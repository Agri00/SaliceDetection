
import struct
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. PARAMETRI MODIFICABILI
# ============================================================

N_CHIRPS = 64
N_SAMPLES = 128
N_RX = 3

CHIRP_PERIOD = 0.0010007125092670321
FRAME_PERIOD = 0.10000739991664886

RADAR_FREQUENCY = 59e9
C = 299792458.0
WAVELENGTH = C / RADAR_FREQUENCY

ADC_SIGNED = True

# Rimozione del clutter statico
REMOVE_CLUTTER = True

# Normalizzazione
NORMALIZATION = "minmax"

# Soglia di binarizzazione
THRESHOLD = 0.75

# Parametri STFT
STFT_WINDOW = 32
STFT_HOP = 16
STFT_NFFT = 64

# Grafici
SAVE_PLOTS = True
PLOT_DPI = 200

# Combinazione ricevitori
# La media viene effettuata sulle potenze dei tre RX.
COMBINE_RX_METHOD = "mean"

# Numero di bin Range positivi utilizzati
N_RANGE_BINS = N_SAMPLES // 2


# ============================================================
# 2. DECODIFICA FILE BIN
# ============================================================

def decode_bin_file(file_path):

    print("\n[1/8] Decodifica file BIN...")

    frames = []
    sequences = []

    expected_payload = (
        N_CHIRPS * N_SAMPLES * N_RX * 12 // 8
    )

    with open(file_path, "rb") as f:

        while True:

            header = f.read(12)

            if len(header) == 0:
                break

            if len(header) != 12:
                raise ValueError(
                    "Header incompleto alla fine del file."
                )

            version, sequence, payload_length = struct.unpack(
                "<III", header
            )

            if payload_length != expected_payload:

                raise ValueError(
                    f"Payload non valido nel record {sequence}: "
                    f"{payload_length} byte, attesi "
                    f"{expected_payload}."
                )

            raw = f.read(payload_length)

            if len(raw) != payload_length:
                raise ValueError(
                    f"Payload incompleto nel record {sequence}."
                )

            raw = np.frombuffer(raw, dtype=np.uint8)

            # Ricostruzione ADC 12 bit packed

            b0 = raw[0::3].astype(np.uint16)
            b1 = raw[1::3].astype(np.uint16)
            b2 = raw[2::3].astype(np.uint16)

            sample0 = (b0 << 4) | (b1 >> 4)

            sample1 = (
                ((b1 & 0x0F) << 8) | b2
            )

            adc = np.empty(
                sample0.size + sample1.size,
                dtype=np.uint16
            )

            adc[0::2] = sample0
            adc[1::2] = sample1

            # Interpretazione complemento a due
            if ADC_SIGNED:

                adc = adc.astype(np.int32)

                adc[adc >= 2048] -= 4096

            else:

                adc = adc.astype(np.float32)

            # Ogni record contiene:
            # 64 chirp x 128 campioni x 3 ricevitori

            adc_frame = adc.reshape(
                N_CHIRPS,
                N_SAMPLES,
                N_RX
            )

            frames.append(adc_frame)
            sequences.append(sequence)

    if len(frames) == 0:
        raise ValueError("Nessun frame trovato nel file.")

    frames = np.asarray(frames, dtype=np.float32)
    sequences = np.asarray(sequences, dtype=np.uint32)

    print(f"Frame decodificati: {len(frames)}")
    print(f"Shape ADC: {frames.shape}")

    print(
        "Ricevitori presenti in ogni frame:",
        frames.shape[3]
    )

    return frames, sequences


# ============================================================
# 3. RANGE FFT
# ============================================================

def calculate_range_fft(adc_data):

    print("\n[2/8] Calcolo Range FFT...")

    # Finestra sui campioni fast-time
    window = np.hanning(N_SAMPLES).astype(np.float32)

    windowed = adc_data * window[None, None, :, None]

    range_fft = np.fft.fft(
        windowed,
        axis=2
    )

    # Conserviamo le frequenze positive
    range_fft = range_fft[:, :, :N_RANGE_BINS, :]

    print("Shape Range FFT:", range_fft.shape)

    return range_fft


# ============================================================
# 4. RIMOZIONE CLUTTER
# ============================================================

def remove_static_clutter(range_fft):

    print("\n[3/8] Rimozione clutter...")

    if not REMOVE_CLUTTER:

        print("Rimozione clutter disattivata.")

        return range_fft

    # Sottrazione della media lungo i chirp.
    # Operazione indipendente per ogni frame,
    # bin Range e ricevitore.

    clutter = np.mean(
        range_fft,
        axis=1,
        keepdims=True
    )

    result = range_fft - clutter

    return result


# ============================================================
# 5. MICRO-DOPPLER
# ============================================================

def calculate_microdoppler(range_fft, sequences):

    print("\n[4/8] Calcolo Micro-Doppler...")

    n_frames = range_fft.shape[0]

    starts = np.arange(
        0,
        N_CHIRPS - STFT_WINDOW + 1,
        STFT_HOP
    )

    n_windows = len(starts)

    # Finestra STFT
    stft_window = np.hanning(
        STFT_WINDOW
    ).astype(np.float32)

    # Asse Doppler
    doppler_hz = np.fft.fftshift(
        np.fft.fftfreq(
            STFT_NFFT,
            d=CHIRP_PERIOD
        )
    )

    velocity = doppler_hz * WAVELENGTH / 2

    microdoppler_power = []
    stft_times = []
    stft_frame_index = []
    stft_sequence = []

    # Tempo relativo alla sequenza dei frame.
    # Gestisce anche l'eventuale rollover uint32.

    seq_delta = (
        sequences.astype(np.uint64)
        - np.uint64(sequences[0])
    ) & np.uint64(0xFFFFFFFF)

    frame_times = (
        seq_delta.astype(np.float64) * FRAME_PERIOD
    )

    for frame_idx in range(n_frames):

        current_frame = range_fft[frame_idx]

        for start in starts:

            end = start + STFT_WINDOW

            # Forma:
            # (chirp, range, RX)

            segment = current_frame[start:end, :, :]

            segment = (
                segment
                * stft_window[:, None, None]
            )

            # FFT lungo la dimensione slow-time
            doppler_fft = np.fft.fft(
                segment,
                n=STFT_NFFT,
                axis=0
            )

            doppler_fft = np.fft.fftshift(
                doppler_fft,
                axes=0
            )

            # Potenza
            power = np.abs(doppler_fft) ** 2

            # ------------------------------------------------
            # COMBINAZIONE DEI TRE RICEVITORI
            # ------------------------------------------------

            if COMBINE_RX_METHOD == "mean":

                # Media non coerente delle potenze RX
                power = np.mean(
                    power,
                    axis=2
                )

            else:

                raise ValueError(
                    "Metodo di combinazione RX non valido."
                )

            # ------------------------------------------------
            # INTEGRAZIONE RANGE
            # ------------------------------------------------

            # Forma iniziale:
            # (Doppler, Range)

            # Forma finale:
            # (Doppler)

            microdoppler = np.sum(
                power,
                axis=1
            )

            microdoppler_power.append(microdoppler)

            # Centro temporale della finestra STFT
            center_chirp = (
                start + (STFT_WINDOW - 1) / 2
            )

            time = (
                frame_times[frame_idx]
                + center_chirp * CHIRP_PERIOD
            )

            stft_times.append(time)

            stft_frame_index.append(frame_idx)

            stft_sequence.append(
                sequences[frame_idx]
            )

    microdoppler_power = np.asarray(
        microdoppler_power,
        dtype=np.float32
    )

    stft_times = np.asarray(
        stft_times,
        dtype=np.float64
    )

    stft_frame_index = np.asarray(
        stft_frame_index,
        dtype=np.int32
    )

    stft_sequence = np.asarray(
        stft_sequence,
        dtype=np.uint32
    )

    print("Shape Micro-Doppler:", microdoppler_power.shape)

    print("Numero di finestre STFT:", len(stft_times))

    print("Ricevitori combinati:", N_RX)

    return (
        microdoppler_power,
        stft_times,
        stft_frame_index,
        stft_sequence,
        doppler_hz,
        velocity
    )


# ============================================================
# 6. CONVERSIONE IN dB
# ============================================================

def convert_to_db(microdoppler_power):

    print("\n[5/8] Conversione in dB...")

    epsilon = 1e-12

    microdoppler_db = 10.0 * np.log10(
        microdoppler_power + epsilon
    )

    return microdoppler_db.astype(np.float32)


# ============================================================
# 7. NORMALIZZAZIONE E BINARIZZAZIONE
# ============================================================

def normalize_microdoppler(microdoppler_db):

    print("\n[6/8] Normalizzazione...")

    global_min = np.min(microdoppler_db)
    global_max = np.max(microdoppler_db)

    if NORMALIZATION == "minmax":

        normalized = (
            microdoppler_db - global_min
        ) / (
            global_max - global_min + 1e-12
        )

    else:

        raise ValueError(
            "Tipo di normalizzazione non riconosciuto."
        )

    return (
        normalized.astype(np.float32),
        float(global_min),
        float(global_max)
    )


def binarize_microdoppler(normalized):

    print("\n[7/8] Binarizzazione...")

    binary = (
        normalized >= THRESHOLD
    ).astype(np.uint8)

    return binary


# ============================================================
# 8. CREAZIONE GRAFICI STATICI
# ============================================================

def create_plot(
    data,
    times,
    velocity,
    output_path,
    title,
    colorbar_label,
    cmap="viridis",
    vmin=None,
    vmax=None
):

    fig, ax = plt.subplots(
        figsize=(18, 7)
    )

    mesh = ax.pcolormesh(
        times,
        velocity,
        data.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)

    ax.set_xlabel("Tempo [s]")

    ax.set_ylabel("Velocità radiale [m/s]")

    fig.colorbar(
        mesh,
        ax=ax,
        label=colorbar_label
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=PLOT_DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Grafico salvato:", output_path)


def save_all_plots(
    microdoppler_db,
    normalized,
    binary,
    times,
    velocity,
    output_dir
):

    print("\n[8/8] Salvataggio grafici...")

    if not SAVE_PLOTS:
        return

    create_plot(
        data=microdoppler_db,
        times=times,
        velocity=velocity,
        output_path=output_dir / "microdoppler_db.png",
        title="Micro-Doppler - Potenza [dB]",
        colorbar_label="Potenza [dB]",
        cmap="viridis"
    )

    create_plot(
        data=normalized,
        times=times,
        velocity=velocity,
        output_path=output_dir / "microdoppler_normalized.png",
        title="Micro-Doppler - Normalizzato",
        colorbar_label="Valore normalizzato",
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    create_plot(
        data=binary,
        times=times,
        velocity=velocity,
        output_path=output_dir / "microdoppler_binary.png",
        title=f"Micro-Doppler - Binarizzato (threshold={THRESHOLD})",
        colorbar_label="Valore binario",
        cmap="gray",
        vmin=0,
        vmax=1
    )


# ============================================================
# 9. SALVATAGGIO DATASET
# ============================================================

def save_dataset(
    output_path,
    microdoppler_power,
    microdoppler_db,
    normalized,
    binary,
    times,
    frame_index,
    sequences,
    doppler_hz,
    velocity,
    original_sequences,
    global_min,
    global_max
):

    print("\nSalvataggio dataset NPZ...")

    np.savez_compressed(

        output_path,

        # Dati principali
        microdoppler_power=microdoppler_power,
        microdoppler_db=microdoppler_db,
        microdoppler_normalized=normalized,
        binary=binary,

        # Informazioni temporali
        stft_time_s=times,
        stft_frame_index=frame_index,
        stft_sequence=sequences,
        sequence=original_sequences,

        # Assi fisici
        doppler_hz=doppler_hz,
        velocity_m_s=velocity,

        # Parametri
        threshold=THRESHOLD,
        normalization=NORMALIZATION,
        global_min=global_min,
        global_max=global_max,

        n_rx=N_RX,
        n_chirps=N_CHIRPS,
        n_samples=N_SAMPLES,

        chirp_period=CHIRP_PERIOD,
        frame_period=FRAME_PERIOD,

        stft_window=STFT_WINDOW,
        stft_hop=STFT_HOP,
        stft_nfft=STFT_NFFT,

        remove_clutter=REMOVE_CLUTTER,
        combine_rx_method=COMBINE_RX_METHOD
    )

    print("Dataset salvato:", output_path)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Pipeline Micro-Doppler FMCW"
    )

    parser.add_argument(
        "input",
        type=str,
        help="Percorso del file BIN"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():

        raise FileNotFoundError(
            f"File non trovato: {input_path}"
        )

    output_dir = (
        input_path.parent
        / (input_path.stem + "_output")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n====================================")
    print(" PIPELINE MICRO-DOPPLER")
    print("====================================")

    print("Input:", input_path)
    print("Output:", output_dir)

    # 1. Decode
    adc_data, sequences = decode_bin_file(
        input_path
    )

    # 2. Range FFT
    range_fft = calculate_range_fft(
        adc_data
    )

    del adc_data

    # 3. Clutter
    range_fft = remove_static_clutter(
        range_fft
    )

    # 4. Micro-Doppler con RX combinati
    (
        microdoppler_power,
        times,
        frame_index,
        stft_sequence,
        doppler_hz,
        velocity
    ) = calculate_microdoppler(
        range_fft,
        sequences
    )

    del range_fft

    # 5. dB
    microdoppler_db = convert_to_db(
        microdoppler_power
    )

    # 6. Normalizzazione
    (
        normalized,
        global_min,
        global_max
    ) = normalize_microdoppler(
        microdoppler_db
    )

    # 7. Binarizzazione
    binary = binarize_microdoppler(
        normalized
    )

    # 8. Grafici
    save_all_plots(
        microdoppler_db,
        normalized,
        binary,
        times,
        velocity,
        output_dir
    )

    # 9. Dataset
    dataset_path = (
        output_dir
        / (input_path.stem + "_microdoppler.npz")
    )

    save_dataset(
        dataset_path,
        microdoppler_power,
        microdoppler_db,
        normalized,
        binary,
        times,
        frame_index,
        stft_sequence,
        doppler_hz,
        velocity,
        sequences,
        global_min,
        global_max
    )

    # Riepilogo
    print("\n====================================")
    print(" ELABORAZIONE COMPLETATA")
    print("====================================")

    print("Frame decodificati:", len(sequences))
    print("Ricevitori combinati:", N_RX)

    print(
        "Shape micro-Doppler:",
        microdoppler_power.shape
    )

    print(
        "Shape dataset binario:",
        binary.shape
    )

    print("Threshold:", THRESHOLD)

    print("\nFile generati:")

    print(" - microdoppler_db.png")
    print(" - microdoppler_normalized.png")
    print(" - microdoppler_binary.png")
    print(" -", dataset_path.name)

    print("\nNessuna GIF generata.")


if __name__ == "__main__":
    main()
