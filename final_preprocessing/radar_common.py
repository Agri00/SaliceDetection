
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAZIONE RADAR
# ============================================================

N_CHIRPS = 64
N_SAMPLES = 128
N_RX = 3

CHIRP_PERIOD = 0.0010007125092670321
FRAME_PERIOD = 0.10000739991664886

RADAR_FREQUENCY = 59e9
SPEED_OF_LIGHT = 299792458.0

ADC_SIGNED = True
EXPECTED_PAYLOAD_SIZE = 36864
HEADER_SIZE = 12

# ============================================================
# RANGE PROCESSING
# ============================================================

N_RANGE_BINS = 64

REMOVE_CLUTTER = True
CLUTTER_ALPHA = 1

# ============================================================
# MICRO-DOPPLER
# ============================================================

STFT_WINDOW = 32
STFT_HOP = 16
STFT_NFFT = 64

# ============================================================
# OUTPUT
# ============================================================

PLOT_DPI = 200


def decode_bin_file(file_path):

    frames = []
    sequences = []

    with open(file_path, "rb") as f:

        frame_number = 0

        while True:

            header = f.read(HEADER_SIZE)

            if len(header) == 0:
                break

            if len(header) != HEADER_SIZE:
                print("Header finale incompleto.")
                break

            version, sequence, payload_length = struct.unpack(
                "<III", header
            )

            if payload_length != EXPECTED_PAYLOAD_SIZE:
                raise ValueError(
                    f"Payload non valido al frame {frame_number}: "
                    f"{payload_length} byte."
                )

            raw = f.read(payload_length)

            if len(raw) != payload_length:
                print("Payload finale incompleto.")
                break

            raw = np.frombuffer(raw, dtype=np.uint8)

            b0 = raw[0::3].astype(np.uint16)
            b1 = raw[1::3].astype(np.uint16)
            b2 = raw[2::3].astype(np.uint16)

            sample0 = (b0 << 4) | (b1 >> 4)
            sample1 = ((b1 & 0x0F) << 8) | b2

            adc = np.empty(
                sample0.size + sample1.size,
                dtype=np.uint16
            )

            adc[0::2] = sample0
            adc[1::2] = sample1

            if ADC_SIGNED:

                adc = adc.astype(np.int32)
                adc[adc >= 2048] -= 4096

            else:

                adc = adc.astype(np.float32)

            expected = N_CHIRPS * N_SAMPLES * N_RX

            if adc.size != expected:
                raise ValueError(
                    f"Numero campioni errato al frame "
                    f"{frame_number}: {adc.size}"
                )

            adc_frame = adc.reshape(
                N_CHIRPS,
                N_SAMPLES,
                N_RX
            )

            frames.append(adc_frame)
            sequences.append(sequence)

            frame_number += 1

    if not frames:
        raise ValueError("Nessun frame valido trovato.")

    adc_data = np.stack(frames, axis=0)

    sequences = np.asarray(
        sequences,
        dtype=np.uint32
    )

    print(f"Frame decodificati: {len(frames)}")

    return adc_data, sequences


def calculate_microdoppler(adc_data, sequences):

    print("Calcolo Range FFT...")

    window_range = np.hanning(N_SAMPLES).astype(
        np.float32
    )

    adc_windowed = (
        adc_data * window_range[None, None, :, None]
    )

    range_fft = np.fft.fft(
        adc_windowed,
        axis=2
    )

    range_fft = range_fft[:, :, :N_RANGE_BINS, :]

    if REMOVE_CLUTTER:

        print("Rimozione clutter statico...")

        mean_clutter = np.mean(
            range_fft,
            axis=1,
            keepdims=True
        )

        range_fft = (
            range_fft - CLUTTER_ALPHA * mean_clutter
        )

    print("Calcolo micro-Doppler...")

    n_frames = range_fft.shape[0]

    starts = np.arange(
        0,
        N_CHIRPS - STFT_WINDOW + 1,
        STFT_HOP
    )

    n_windows = len(starts)

    total_windows = n_frames * n_windows

    microdoppler_power = np.zeros(
        (total_windows, STFT_NFFT),
        dtype=np.float64
    )

    frame_indices = np.zeros(
        total_windows,
        dtype=np.int32
    )

    seq_delta = (
        sequences.astype(np.uint64)
        - np.uint64(sequences[0])
    ) & np.uint64(0xFFFFFFFF)

    frame_times = (
        seq_delta.astype(np.float64) * FRAME_PERIOD
    )

    time_values = np.zeros(
        total_windows,
        dtype=np.float64
    )

    window_doppler = np.hanning(
        STFT_WINDOW
    )

    output_index = 0

    for frame_idx in range(n_frames):

        current_frame = range_fft[frame_idx]

        for start in starts:

            segment = current_frame[
                start:start + STFT_WINDOW,
                :, :
            ]

            segment = (
                segment * window_doppler[:, None, None]
            )

            doppler_fft = np.fft.fft(
                segment,
                n=STFT_NFFT,
                axis=0
            )

            doppler_fft = np.fft.fftshift(
                doppler_fft,
                axes=0
            )

            power = np.abs(doppler_fft) ** 2

            # Media dei tre ricevitori

            power = np.mean(power, axis=2)

            # Somma sui bin di range

            microdoppler_power[output_index] = np.sum(
                power,
                axis=1
            )

            frame_indices[output_index] = frame_idx

            center_chirp = (
                start + (STFT_WINDOW - 1) / 2
            )

            time_values[output_index] = (
                frame_times[frame_idx]
                + center_chirp * CHIRP_PERIOD
            )

            output_index += 1

    print(
        "Micro-Doppler shape:",
        microdoppler_power.shape
    )

    return (
        microdoppler_power,
        frame_indices,
        time_values
    )


def convert_to_db(power):

    return 10.0 * np.log10(
        power + 1e-12
    )


def normalize_minmax(microdoppler_db):

    minimum = np.min(microdoppler_db)
    maximum = np.max(microdoppler_db)

    normalized = (
        microdoppler_db - minimum
    ) / (
        maximum - minimum + 1e-12
    )

    return normalized


def calculate_velocity_axis():

    wavelength = (
        SPEED_OF_LIGHT / RADAR_FREQUENCY
    )

    doppler_frequency = np.fft.fftshift(
        np.fft.fftfreq(
            STFT_NFFT,
            d=CHIRP_PERIOD
        )
    )

    velocity = (
        doppler_frequency * wavelength / 2.0
    )

    return velocity


def save_plot(
    matrix,
    time_values,
    velocity,
    title,
    filename,
    output_dir,
    colorbar_label,
    cmap
):

    plt.figure(figsize=(14, 6))

    extent = [
        time_values[0],
        time_values[-1],
        velocity[0],
        velocity[-1]
    ]

    plt.imshow(
        matrix.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest"
    )

    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocità (m/s)")
    plt.title(title)

    plt.colorbar(label=colorbar_label)

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        path,
        dpi=PLOT_DPI,
        bbox_inches="tight"
    )

    plt.close()

    print("Salvato:", path)
