
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DREAMHAT+ / BGT60TR13C - PIPELINE A STEP

STEP 1: Decodifica ADC + Range FFT + STFT micro-Doppler
        -> microdoppler.gif (RX1, RX2, RX3)

STEP 2: Processing delle mappe micro-Doppler
        -> ampiezza, normalizzazione, resize 24x24
        -> grayscale.gif

STEP 3: Binarizzazione
        -> binary.gif

STEP 4: Salvataggio dataset finale per BNN
        -> dreamhat_nn_dataset.npz

Dopo ogni GIF il programma si ferma per consentire
il controllo visivo. INVIO = continua, q = interrompi.

Dipendenze:
    py -m pip install numpy matplotlib pillow
"""

from pathlib import Path
import struct

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image


# ============================================================
# 1. PARAMETRI RADAR
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

N_WINDOWS = 3


# ============================================================
# 3. PARAMETRI DATASET
# ============================================================

IMAGE_SIZE = 24

THRESHOLD = 0.10

# "per_map" = normalizzazione indipendente di ogni mappa
# "global"  = normalizzazione globale dell'intera registrazione

NORMALIZATION_MODE = "global"

FPS = 10


# ============================================================
# 4. FUNZIONI DI SUPPORTO
# ============================================================

def ask_to_continue(message):

    print("\n" + message)

    answer = input(
        "Premi INVIO per continuare oppure scrivi q per uscire: "
    )

    if answer.strip().lower() in ("q", "quit", "exit"):

        print("\nElaborazione interrotta.")
        print("I file già creati sono stati conservati.")

        raise SystemExit(0)


# ============================================================
# 5. DECODIFICA ADC 12 BIT
# ============================================================

def unpack_12bit_fifo(payload):

    if len(payload) % 3 != 0:
        raise ValueError("Payload non divisibile per 3.")

    b = np.frombuffer(
        payload,
        dtype=np.uint8
    ).reshape(-1, 3).astype(np.uint16)

    sample0 = (
        (b[:, 0] << 4)
        | (b[:, 1] >> 4)
    )

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

    print("\n[1/4] DECODIFICA ADC")

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
            "<III",
            raw,
            offset
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

    sequence = np.asarray(
        sequence_numbers,
        dtype=np.uint32
    )

    expected_seq = np.arange(
        sequence[0],
        sequence[0] + len(sequence),
        dtype=np.uint32
    )

    if not np.array_equal(sequence, expected_seq):

        raise ValueError(
            "I numeri di sequenza non sono consecutivi. "
            "Possibili frame persi."
        )

    adc = np.stack(frames, axis=0)

    frame_time_s = (
        (sequence - sequence[0]).astype(np.float64)
        * FRAME_PERIOD_S
    )

    print("Frame decodificati:", len(sequence))
    print("Shape ADC:", adc.shape)

    return adc, sequence, frame_time_s


# ============================================================
# 6. RANGE FFT E MICRO-DOPPLER
# ============================================================

def calculate_microdoppler(adc, frame_time_s):

    print("\nCalcolo Range FFT e STFT...")

    n_frames = adc.shape[0]

    # --------------------------------------------------------
    # Conversione in float
    # --------------------------------------------------------

    x = adc.astype(np.float32)

    # Rimozione media fast-time
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

    # Shape:
    # frame, finestra, range, Doppler, RX

    power_per_rx = np.empty(
        (
            n_frames,
            N_WINDOWS,
            64,
            64,
            N_RX
        ),
        dtype=np.float32
    )

    for w in range(N_WINDOWS):

        start = w * STFT_HOP
        end = start + STFT_WINDOW

        segment = range_fft[:, start:end, :, :]

        segment *= stft_window[
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

        # frame, Doppler, Range, RX
        # ->
        # frame, Range, Doppler, RX

        doppler_fft = np.transpose(
            doppler_fft,
            (0, 2, 1, 3)
        )

        power_per_rx[:, w] = (
            np.abs(doppler_fft) ** 2
        ).astype(np.float32)

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
    # Tempi centrali delle finestre STFT
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

    print("Shape potenza per RX:", power_per_rx.shape)

    return (
        power_per_rx,
        range_m,
        doppler_hz,
        velocity_m_s,
        stft_time_s
    )


# ============================================================
# 7. GIF MICRO-DOPPLER: RX1, RX2, RX3
# ============================================================

def save_microdoppler_gif(
    power_per_rx,
    range_m,
    velocity_m_s,
    stft_time_s,
    sequence,
    output_path
):

    print("\n[2/4] CREAZIONE GIF MICRO-DOPPLER")

    n_frames, n_windows = power_per_rx.shape[:2]

    # Ordine:
    # frame, finestra, RX, range, Doppler

    data = np.transpose(
        power_per_rx,
        (0, 1, 4, 2, 3)
    )

    data = data.reshape(
        -1,
        N_RX,
        64,
        64
    )

    times = stft_time_s.reshape(-1)

    frame_indices = np.repeat(
        np.arange(n_frames),
        n_windows
    )

    window_indices = np.tile(
        np.arange(n_windows),
        n_frames
    )

    sequence_per_image = sequence[frame_indices]

    data_db = 10 * np.log10(
        np.maximum(data, 1e-12)
    )

    vmin = float(np.percentile(data_db, 5))
    vmax = float(np.percentile(data_db, 99.5))

    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
        sharex=True,
        sharey=True
    )

    images = []

    for rx in range(N_RX):

        im = axes[rx].imshow(
            data_db[0, rx],
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

        axes[rx].set_title(f"RX{rx + 1}")

        axes[rx].set_xlabel(
            "Velocità radiale (m/s)"
        )

        axes[rx].set_ylabel(
            "Distanza (m)"
        )

        images.append(im)

    fig.colorbar(
        images[0],
        ax=list(axes),
        label="Potenza (dB)"
    )

    title = fig.suptitle("Micro-Doppler")

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    def update(i):

        for rx in range(N_RX):
            images[rx].set_data(data_db[i, rx])

        title.set_text(
            f"Frame originale {sequence_per_image[i]} | "
            f"Finestra STFT {window_indices[i] + 1}/3 | "
            f"t = {times[i]:.4f} s"
        )

        return images + [title]

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
        dpi=70
    )

    plt.close(fig)

    print("Salvata:", output_path)
    print("Numero immagini GIF:", len(data))


# ============================================================
# 8. PROCESSING GRAYSCALE
# ============================================================

def build_grayscale_dataset(power_per_rx):

    print("\n[3/4] PROCESSING GRAYSCALE")

    # Media delle potenze dei tre ricevitori
    power = power_per_rx.mean(
        axis=-1
    ).astype(np.float32)

    n_frames, n_windows = power.shape[:2]

    n_images = n_frames * n_windows

    # Conversione potenza -> ampiezza
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
        print("Minimo:", global_min)
        print("Massimo:", global_max)

    elif NORMALIZATION_MODE == "per_map":

        global_min = None
        global_max = None

        print("Normalizzazione per singola mappa")

    else:

        raise ValueError(
            "NORMALIZATION_MODE deve essere "
            "'per_map' oppure 'global'."
        )

    X_gray = np.empty(
        (n_images, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32
    )

    frame_index = np.repeat(
        np.arange(n_frames),
        n_windows
    )

    window_index = np.tile(
        np.arange(n_windows),
        n_frames
    )

    # --------------------------------------------------------
    # Normalizzazione e resize
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

        gray_uint8 = np.round(
            gray * 255
        ).astype(np.uint8)

        pil_image = Image.fromarray(
            gray_uint8
        )

        pil_image = pil_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        X_gray[i] = (
            np.asarray(
                pil_image,
                dtype=np.float32
            ) / 255.0
        )

    print("Shape grayscale:", X_gray.shape)

    return X_gray, frame_index, window_index


# ============================================================
# 9. GIF GRAYSCALE E BINARIA
# ============================================================

def save_image_gif(
    images,
    times,
    frame_index,
    window_index,
    sequence,
    output_path,
    title_text,
    binary=False
):

    n_images = len(images)

    print(f"\nCreazione GIF {title_text}...")

    sequence_per_image = sequence[frame_index]

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    image = ax.imshow(
        images[0],
        cmap="gray",
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=1
    )

    # Assi pixel 0-23
    ax.set_xticks(np.arange(IMAGE_SIZE))
    ax.set_yticks(np.arange(IMAGE_SIZE))

    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

    ax.set_xlim(-0.5, IMAGE_SIZE - 0.5)
    ax.set_ylim(-0.5, IMAGE_SIZE - 0.5)

    title = ax.set_title(title_text)

    fig.tight_layout()

    def update(i):

        image.set_data(images[i])

        title.set_text(
            f"{title_text} | "
            f"Frame {sequence_per_image[i]} | "
            f"Finestra {window_index[i] + 1}/3 | "
            f"t = {times[i]:.4f} s"
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
        dpi=80
    )

    plt.close(fig)

    print("Salvata:", output_path)
    print("Numero immagini GIF:", n_images)


# ============================================================
# 10. BINARIZZAZIONE
# ============================================================

def binarize_dataset(X_gray):

    print("\n[4/4] BINARIZZAZIONE")

    print("Threshold:", THRESHOLD)

    X_bin = (
        X_gray >= THRESHOLD
    ).astype(np.uint8)

    print("Shape binaria:", X_bin.shape)

    return X_bin


# ============================================================
# 11. PIPELINE PRINCIPALE
# ============================================================

def main():

    print("\n======================================")
    print(" DREAMHAT+ PROCESSING A STEP")
    print("======================================")

    print("\nParametri:")
    print("Threshold:", THRESHOLD)
    print("Dimensione immagini:", IMAGE_SIZE)
    print("Normalizzazione:", NORMALIZATION_MODE)

    # --------------------------------------------------------
    # Selezione file BIN
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

    # ========================================================
    # STEP 1 - DECODIFICA E MICRO-DOPPLER
    # ========================================================

    adc, sequence, frame_time_s = decode_recording(
        source_path
    )

    (
        power_per_rx,
        range_m,
        doppler_hz,
        velocity_m_s,
        stft_time_s
    ) = calculate_microdoppler(
        adc,
        frame_time_s
    )

    # L'ADC non serve più dopo il calcolo micro-Doppler
    del adc

    # ========================================================
    # STEP 2 - GIF MICRO-DOPPLER
    # ========================================================

    micro_gif = output_dir / "microdoppler.gif"

    save_microdoppler_gif(
        power_per_rx,
        range_m,
        velocity_m_s,
        stft_time_s,
        sequence,
        micro_gif
    )

    # ========================================================
    # STEP 3 - GRAYSCALE
    # ========================================================

    X_gray, frame_index, window_index = (
        build_grayscale_dataset(power_per_rx)
    )

    time_s = stft_time_s[
        frame_index,
        window_index
    ]

    # La potenza per RX non serve più
    del power_per_rx

    # ========================================================
    # STEP 4 - GIF GRAYSCALE
    # ========================================================

    gray_gif = output_dir / "grayscale.gif"

    save_image_gif(
        X_gray,
        time_s,
        frame_index,
        window_index,
        sequence,
        gray_gif,
        "Grayscale"
    )

    # ========================================================
    # STEP 5 - BINARIZZAZIONE
    # ========================================================

    X_bin = binarize_dataset(X_gray)

    # La grayscale non serve più per il dataset finale
    del X_gray

    # ========================================================
    # STEP 6 - GIF BINARIA
    # ========================================================

    binary_gif = output_dir / "binary.gif"

    save_image_gif(
        X_bin.astype(np.float32),
        time_s,
        frame_index,
        window_index,
        sequence,
        binary_gif,
        f"Binary (threshold={THRESHOLD})",
        binary=True
    )

    # ========================================================
    # STEP 7 - DATASET FINALE PER LA BNN
    # ========================================================

    print("\nSalvataggio dataset per rete neurale...")

    labels = np.full(
        len(X_bin),
        -1,
        dtype=np.int64
    )

    sequence_per_image = sequence[frame_index]

    dataset_path = (
        output_dir / "dreamhat_nn_dataset.npz"
    )

    np.savez_compressed(

        dataset_path,

        # Input della rete binaria
        X_bin=X_bin,

        # Etichette da assegnare
        labels=labels,

        # Riferimenti temporali
        frame_index=frame_index,
        window_index=window_index,
        sequence=sequence_per_image,
        time_s=time_s,

        # Parametri di elaborazione
        threshold=np.float32(THRESHOLD),
        image_size=np.int32(IMAGE_SIZE),
        normalization_mode=np.array(NORMALIZATION_MODE),

        stft_window=np.int32(STFT_WINDOW),
        stft_hop=np.int32(STFT_HOP),
        n_doppler_fft=np.int32(N_DOPPLER_FFT),

        chirp_period_s=np.float64(CHIRP_PERIOD_S),
        frame_period_s=np.float64(FRAME_PERIOD_S),

        # Assi fisici
        range_m=range_m,
        doppler_hz=doppler_hz,
        velocity_m_s=velocity_m_s
    )

    # ========================================================
    # RIEPILOGO
    # ========================================================

    print("\n======================================")
    print(" ELABORAZIONE COMPLETATA")
    print("======================================")

    print("\nFrame radar originali:", len(sequence))
    print("Mappe elaborate:", len(X_bin))
    print("Shape dataset:", X_bin.shape)

    print("\nFile creati:")

    print(" - microdoppler.gif")
    print(" - grayscale.gif")
    print(" - binary.gif")
    print(" - dreamhat_nn_dataset.npz")

    print("\nDataset finale:")
    print(dataset_path)

    print("\nCartella:")
    print(output_dir.resolve())

    print("\nATTENZIONE:")
    print("Le label sono inizializzate a -1.")
    print("Devono essere assegnate prima del training.")


# ============================================================
# 12. AVVIO
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print("\nElaborazione interrotta dall'utente.")

    except Exception as error:

        print("\nERRORE:")
        print(error)

        raise
