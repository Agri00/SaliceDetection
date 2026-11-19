
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PIPELINE COMPLETA DREAMHAT+ / BGT60TR13C

1. Decode del file BIN
2. Range FFT
3. Rimozione clutter
4. STFT micro-Doppler
5. Normalizzazione
6. Resize 64x64 -> 24x24
7. Binarizzazione
8. Salvataggio dataset NPZ
9. Creazione di tre GIF:
   - microdoppler.gif
   - grayscale.gif
   - binary.gif

Avvio:
    py dreamhat_pipeline.py

Dipendenze:
    py -m pip install numpy matplotlib pillow
"""

from pathlib import Path
import struct
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image


# ============================================================
# 1. PARAMETRI DI ACQUISIZIONE
# ============================================================

N_CHIRPS = 64
N_SAMPLES = 128
N_RX = 3

FS_HZ = 2_000_000.0

CHIRP_PERIOD_S = 0.0010007125092670321
FRAME_PERIOD_S = 0.10000739991664886

F_START_HZ = 58e9
F_END_HZ = 60e9

C_M_S = 299792458.0
FC_HZ = (F_START_HZ + F_END_HZ) / 2

HEADER_BYTES = 12

EXPECTED_SAMPLES = N_CHIRPS * N_SAMPLES * N_RX
EXPECTED_PAYLOAD_BYTES = (EXPECTED_SAMPLES // 2) * 3


# ============================================================
# 2. PARAMETRI MICRO-DOPPLER
# ============================================================

STFT_WINDOW = 32
STFT_HOP = 16
N_DOPPLER_FFT = 64

# Tre finestre per ogni frame:
# 0-31, 16-47, 32-63

N_WINDOWS = 3


# ============================================================
# 3. PARAMETRI DATASET BINARIO
# ============================================================

IMAGE_SIZE = 24

THRESHOLD = 0.15

# Possibili valori:
# "per_map" = normalizzazione separata per ogni immagine
# "global"  = normalizzazione sull'intero dataset

NORMALIZATION_MODE = "per_map"

# Velocità delle GIF
FPS = 10


# ============================================================
# 4. DECODING ADC 12 BIT
# ============================================================

def unpack_12bit_fifo(payload):

    if len(payload) % 3 != 0:
        raise ValueError("Payload non divisibile per 3.")

    b = np.frombuffer(
        payload,
        dtype=np.uint8
    ).reshape(-1, 3).astype(np.uint16)

    sample0 = (b[:, 0] << 4) | (b[:, 1] >> 4)

    sample1 = (
        ((b[:, 1] & 0x0F) << 8)
        | b[:, 2]
    )

    out = np.empty(
        sample0.size * 2,
        dtype=np.uint16
    )

    out[0::2] = sample0
    out[1::2] = sample1

    return out


def decode_recording(path):

    print("\n[1/6] Decodifica registrazione ADC...")

    raw = path.read_bytes()

    offset = 0

    frames = []
    sequence_numbers = []

    while offset < len(raw):

        if len(raw) - offset < HEADER_BYTES:
            raise ValueError(
                f"Header incompleto all'offset {offset}."
            )

        version, seq, payload_len = struct.unpack_from(
            "<III", raw, offset
        )

        if version != 0:
            raise ValueError(
                f"Versione formato inattesa: {version}"
            )

        if payload_len != EXPECTED_PAYLOAD_BYTES:
            raise ValueError(
                f"Payload frame {seq}: {payload_len} byte. "
                f"Attesi {EXPECTED_PAYLOAD_BYTES}."
            )

        start = offset + HEADER_BYTES
        end = start + payload_len

        if end > len(raw):
            raise ValueError(
                f"Frame {seq} incompleto."
            )

        payload = raw[start:end]

        samples = unpack_12bit_fifo(payload)

        if samples.size != EXPECTED_SAMPLES:
            raise ValueError(
                f"Numero campioni errato nel frame {seq}."
            )

        frame = samples.reshape(
            N_CHIRPS,
            N_SAMPLES,
            N_RX
        )

        frames.append(frame)
        sequence_numbers.append(seq)

        offset = end

    if not frames:
        raise ValueError("Nessun frame trovato.")

    seq = np.asarray(
        sequence_numbers,
        dtype=np.uint32
    )

    expected_seq = np.arange(
        seq[0],
        seq[0] + len(seq),
        dtype=np.uint32
    )

    if not np.array_equal(seq, expected_seq):
        raise ValueError(
            "I numeri di sequenza non sono consecutivi. "
            "Possibili frame persi."
        )

    adc = np.stack(frames, axis=0)

    frame_time_s = (
        (seq - seq[0]).astype(np.float64)
        * FRAME_PERIOD_S
    )

    print("Frame decodificati:", len(seq))
    print("Shape ADC:", adc.shape)

    return adc, seq, frame_time_s


# ============================================================
# 5. RANGE FFT E MICRO-DOPPLER
# ============================================================

def calculate_microdoppler(adc, frame_time_s):

    print("\n[2/6] Calcolo Range FFT e micro-Doppler...")

    n_frames = adc.shape[0]

    # --------------------------------------------------------
    # Conversione in float
    # --------------------------------------------------------

    x = adc.astype(np.float32)

    # Rimozione componente media fast-time
    x -= x.mean(
        axis=2,
        keepdims=True
    )

    # --------------------------------------------------------
    # Range FFT
    # --------------------------------------------------------

    range_window = np.hanning(
        N_SAMPLES
    ).astype(np.float32)

    x *= range_window[
        None, None, :, None
    ]

    range_fft = np.fft.fft(
        x,
        n=N_SAMPLES,
        axis=2
    )

    # Manteniamo i 64 bin positivi
    range_fft = range_fft[:, :, :64, :]

    range_fft = range_fft.astype(np.complex64)

    # --------------------------------------------------------
    # Clutter removal
    # --------------------------------------------------------

    range_fft -= range_fft.mean(
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------------
    # STFT lungo slow-time
    # --------------------------------------------------------

    stft_window = np.hanning(
        STFT_WINDOW
    ).astype(np.float32)

    complex_per_rx = np.empty(
        (
            n_frames,
            N_WINDOWS,
            64,
            64,
            N_RX
        ),
        dtype=np.complex64
    )

    for w in range(N_WINDOWS):

        start = w * STFT_HOP
        end = start + STFT_WINDOW

        segment = range_fft[:, start:end, :, :]

        segment = segment * stft_window[
            None, :, None, None
        ]

        doppler_fft = np.fft.fft(
            segment,
            n=N_DOPPLER_FFT,
            axis=1
        )

        doppler_fft = np.fft.fftshift(
            doppler_fft,
            axes=1
        )

        # Da:
        # frame, Doppler, Range, RX
        #
        # A:
        # frame, Range, Doppler, RX

        doppler_fft = np.transpose(
            doppler_fft,
            (0, 2, 1, 3)
        )

        complex_per_rx[:, w] = doppler_fft

    # --------------------------------------------------------
    # Potenza per RX e combinazione
    # --------------------------------------------------------

    power_per_rx = (
        np.abs(complex_per_rx) ** 2
    ).astype(np.float32)

    power_combined = power_per_rx.mean(
        axis=-1
    ).astype(np.float32)

    # Shape:
    # (frame, finestra, range, Doppler)

    # --------------------------------------------------------
    # Assi fisici
    # --------------------------------------------------------

    doppler_hz = np.fft.fftshift(
        np.fft.fftfreq(
            N_DOPPLER_FFT,
            d=CHIRP_PERIOD_S
        )
    )

    wavelength = C_M_S / FC_HZ

    velocity_m_s = (
        doppler_hz * wavelength / 2
    )

    bandwidth = F_END_HZ - F_START_HZ

    range_resolution = (
        C_M_S / (2 * bandwidth)
    )

    range_m = (
        np.arange(64) * range_resolution
    )

    # --------------------------------------------------------
    # Tempi delle finestre STFT
    # --------------------------------------------------------

    stft_time_s = np.empty(
        (n_frames, N_WINDOWS),
        dtype=np.float64
    )

    for w in range(N_WINDOWS):

        start = w * STFT_HOP

        center_chirp = (
            start + (STFT_WINDOW - 1) / 2
        )

        stft_time_s[:, w] = (
            frame_time_s
            + center_chirp * CHIRP_PERIOD_S
        )

    print("Shape micro-Doppler:", power_combined.shape)

    return (
        power_combined,
        power_per_rx,
        complex_per_rx,
        range_m,
        doppler_hz,
        velocity_m_s,
        stft_time_s
    )


# ============================================================
# 6. NORMALIZZAZIONE, RESIZE E BINARIZZAZIONE
# ============================================================

def build_binary_dataset(power):

    print("\n[3/6] Creazione dataset grayscale e binario...")

    n_frames, n_windows, _, _ = power.shape

    n_images = n_frames * n_windows

    # Convertiamo la potenza in ampiezza
    amplitude = np.sqrt(
        np.maximum(power, 0)
    )

    # --------------------------------------------------------
    # Normalizzazione
    # --------------------------------------------------------

    if NORMALIZATION_MODE == "global":

        global_min = float(amplitude.min())
        global_max = float(amplitude.max())

        print("Normalizzazione globale")
        print("Minimo globale:", global_min)
        print("Massimo globale:", global_max)

    elif NORMALIZATION_MODE == "per_map":

        global_min = None
        global_max = None

        print("Normalizzazione per singola mappa")

    else:
        raise ValueError(
            "NORMALIZATION_MODE deve essere "
            "'global' oppure 'per_map'."
        )

    X_gray = np.empty(
        (n_images, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32
    )

    X_bin = np.empty(
        (n_images, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    frame_index = np.repeat(
        np.arange(n_frames),
        n_windows
    )

    window_index = np.tile(
        np.arange(n_windows),
        n_frames
    )

    time_s = np.empty(
        n_images,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Elaborazione delle immagini
    # --------------------------------------------------------

    for i in range(n_images):

        f = frame_index[i]
        w = window_index[i]

        img = amplitude[f, w]

        if NORMALIZATION_MODE == "per_map":

            img_min = float(img.min())
            img_max = float(img.max())

        else:

            img_min = global_min
            img_max = global_max

        gray = (
            img - img_min
        ) / (
            img_max - img_min + 1e-12
        )

        gray = np.clip(
            gray,
            0,
            1
        )

        # Conversione uint8 per PIL
        gray_uint8 = np.round(
            gray * 255
        ).astype(np.uint8)

        pil_image = Image.fromarray(
            gray_uint8
        )

        # Resize 64x64 -> 24x24
        pil_image = pil_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        gray_small = (
            np.asarray(
                pil_image,
                dtype=np.float32
            ) / 255.0
        )

        X_gray[i] = gray_small

        # Binarizzazione
        X_bin[i] = (
            gray_small >= THRESHOLD
        ).astype(np.uint8)

        time_s[i] = 0.0  # Sostituito subito dopo

    return X_gray, X_bin, frame_index, window_index


# ============================================================
# 7. CREAZIONE GIF
# ============================================================

def save_microdoppler_gif(
    power,
    range_m,
    velocity_m_s,
    time_s,
    output_path
):

    print("\n[5/6] Creazione GIF micro-Doppler...")

    n_frames, n_windows = power.shape[:2]

    data = power.reshape(
        -1,
        power.shape[-2],
        power.shape[-1]
    )

    times = time_s.reshape(-1)

    data_db = 10 * np.log10(
        np.maximum(data, 1e-12)
    )

    vmin = float(np.percentile(data_db, 5))
    vmax = float(np.percentile(data_db, 99.5))

    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(
        data_db[0],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="jet",
        extent=[
            velocity_m_s[0],
            velocity_m_s[-1],
            range_m[0],
            range_m[-1]
        ],
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel("Velocità radiale (m/s)")
    ax.set_ylabel("Distanza (m)")

    title = ax.set_title(
        f"Micro-Doppler - t = {times[0]:.3f} s"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Potenza (dB)"
    )

    fig.tight_layout()

    def update(i):

        image.set_data(data_db[i])

        title.set_text(
            f"Micro-Doppler - "
            f"frame {i + 1}/{len(data)} - "
            f"t = {times[i]:.3f} s"
        )

        return image, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(data),
        interval=1000 / FPS,
        blit=False
    )

    animation.save(
        str(output_path),
        writer=PillowWriter(fps=FPS),
        dpi=50
    )

    plt.close(fig)

    print("Salvata:", output_path)


def save_image_gif(
    images,
    times,
    output_path,
    title_text,
    binary=False
):

    n_images = len(images)

    print(f"\nCreazione GIF {title_text}...")

    fig, ax = plt.subplots(figsize=(5, 4))

    cmap = "gray"

    image = ax.imshow(
        images[0],
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=1
    )

    title = ax.set_title(
        f"{title_text} - t = {times[0]:.3f} s"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    def update(i):

        image.set_data(images[i])

        title.set_text(
            f"{title_text} - "
            f"immagine {i + 1}/{n_images} - "
            f"t = {times[i]:.3f} s"
        )

        return image, title

    animation = FuncAnimation(
        fig,
        update,
        frames=n_images,
        interval=1000 / FPS,
        blit=False
    )

    animation.save(
        str(output_path),
        writer=PillowWriter(fps=FPS),
        dpi=50
    )

    plt.close(fig)

    print("Salvata:", output_path)


# ============================================================
# 8. PIPELINE PRINCIPALE
# ============================================================

def main():

    print("\n====================================")
    print(" DREAMHAT+ COMPLETE PROCESSING")
    print("====================================")

    # --------------------------------------------------------
    # Selezione file
    # --------------------------------------------------------

    raw_path = input(
        "\nInserisci il percorso completo del file BIN:\n"
    ).strip().strip('"').strip("'")

    source_path = Path(raw_path).expanduser()

    if not source_path.is_file():
        raise FileNotFoundError(
            f"File non trovato: {source_path}"
        )

    # --------------------------------------------------------
    # Cartella output
    # --------------------------------------------------------

    output_dir = source_path.parent / (
        source_path.stem + "_processed"
    )

    counter = 2

    while output_dir.exists():

        output_dir = source_path.parent / (
            source_path.stem + f"_processed_{counter}"
        )

        counter += 1

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    print("\nCartella output:")
    print(output_dir)

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    adc, sequence, frame_time_s = decode_recording(
        source_path
    )

    # --------------------------------------------------------
    # Micro-Doppler
    # --------------------------------------------------------

    (
        power,
        power_per_rx,
        complex_per_rx,
        range_m,
        doppler_hz,
        velocity_m_s,
        stft_time_s
    ) = calculate_microdoppler(
        adc,
        frame_time_s
    )

    # --------------------------------------------------------
    # Dataset grayscale e binario
    # --------------------------------------------------------

    X_gray, X_bin, frame_index, window_index = (
        build_binary_dataset(power)
    )

    time_s = stft_time_s[
        frame_index,
        window_index
    ]

    labels = np.full(
        len(X_bin),
        -1,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Salvataggio dataset completo
    # --------------------------------------------------------

    print("\n[4/6] Salvataggio dataset NPZ...")

    dataset_path = (
        output_dir / "dreamhat_full_dataset.npz"
    )

    np.savez_compressed(
        dataset_path,

        # Dati decodificati
        adc=adc,
        sequence=sequence,
        frame_time_s=frame_time_s,

        # Micro-Doppler
        micro_doppler_power=power,
        micro_doppler_power_per_rx=power_per_rx,
        micro_doppler_complex_per_rx=complex_per_rx,

        # Assi
        range_m=range_m,
        doppler_hz=doppler_hz,
        velocity_m_s=velocity_m_s,

        # Tempi
        stft_time_s=stft_time_s,
        time_s=time_s,

        # Dataset per rete neurale
        X_gray=X_gray,
        X_bin=X_bin,
        labels=labels,

        # Indici
        frame_index=frame_index,
        window_index=window_index,

        # Parametri
        threshold=THRESHOLD,
        normalization_mode=NORMALIZATION_MODE,
        image_size=IMAGE_SIZE,
        stft_window=STFT_WINDOW,
        stft_hop=STFT_HOP,
        n_doppler_fft=N_DOPPLER_FFT,
        chirp_period_s=CHIRP_PERIOD_S,
        frame_period_s=FRAME_PERIOD_S
    )

    print("Dataset salvato:", dataset_path)

    # --------------------------------------------------------
    # GIF micro-Doppler
    # --------------------------------------------------------

    save_microdoppler_gif(
        power,
        range_m,
        velocity_m_s,
        stft_time_s,
        output_dir / "microdoppler.gif"
    )

    # --------------------------------------------------------
    # GIF grayscale
    # --------------------------------------------------------

    save_image_gif(
        X_gray,
        time_s,
        output_dir / "grayscale.gif",
        "Grayscale"
    )

    # --------------------------------------------------------
    # GIF binaria
    # --------------------------------------------------------

    save_image_gif(
        X_bin,
        time_s,
        output_dir / "binary.gif",
        "Binary"
    )

    # --------------------------------------------------------
    # Fine
    # --------------------------------------------------------

    print("\n====================================")
    print(" ELABORAZIONE COMPLETATA")
    print("====================================")

    print("\nFrame originali:", len(sequence))
    print("Numero mappe:", len(X_bin))

    print("Shape ADC:", adc.shape)
    print("Shape micro-Doppler:", power.shape)
    print("Shape grayscale:", X_gray.shape)
    print("Shape binary:", X_bin.shape)

    print("\nFile creati:")
    print(" - dreamhat_full_dataset.npz")
    print(" - microdoppler.gif")
    print(" - grayscale.gif")
    print(" - binary.gif")

    print("\nCartella:", output_dir.resolve())


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nERRORE:")
        print(error)

        raise
