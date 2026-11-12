
import numpy as np
import argparse
from pathlib import Path

# Parametri del sensore
CHIRP_PERIOD = 0.0010007125092670321
FRAME_PERIOD = 0.10000739991664886

# Configurazione STFT iniziale
STFT_WINDOW = 32
STFT_HOP = 16
STFT_NFFT = 64


def hann_window(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def process_adc(adc, frame_time):
    """
    adc: (frames, chirps, samples, RX)

    Restituisce:
      micro_doppler: (frames, finestre, doppler, range)
      micro_doppler_rx: (frames, finestre, doppler, range, RX)
    """

    n_frames, n_chirps, n_samples, n_rx = adc.shape

    # --------------------------------------------------
    # 1. RANGE FFT
    # --------------------------------------------------

    x = adc.astype(np.float32)

    # Rimozione componente continua fast-time
    x -= x.mean(axis=2, keepdims=True)

    # Finestra Hann sui campioni del chirp
    x *= hann_window(n_samples)[None, None, :, None]

    # FFT complessa
    range_fft = np.fft.fft(x, axis=2)

    # Manteniamo le frequenze positive
    range_fft = range_fft[:, :, :n_samples // 2, :]

    # --------------------------------------------------
    # 2. RIMOZIONE CLUTTER STATICO
    # --------------------------------------------------

    # Sottrazione della media lungo gli slow-time samples
    slow = range_fft - range_fft.mean(axis=1, keepdims=True)

    # --------------------------------------------------
    # 3. STFT SULLO SLOW-TIME
    # --------------------------------------------------

    starts = range(
        0,
        n_chirps - STFT_WINDOW + 1,
        STFT_HOP
    )

    window = hann_window(STFT_WINDOW)
    stft_results = []

    for start in starts:

        segment = slow[
            :,
            start:start + STFT_WINDOW,
            :,
            :
        ]

        segment = segment * window[None, :, None, None]

        # FFT lungo la dimensione temporale dei chirp
        spectrum = np.fft.fft(
            segment,
            n=STFT_NFFT,
            axis=1
        )

        # Centratura dello spettro Doppler
        spectrum = np.fft.fftshift(spectrum, axes=1)

        stft_results.append(spectrum)

    # Shape:
    # (frames, finestre STFT, Doppler, range, RX)
    stft_complex = np.stack(stft_results, axis=1)

    # --------------------------------------------------
    # 4. POTENZA
    # --------------------------------------------------

    power_rx = np.abs(stft_complex) ** 2

    # Media non coerente dei tre ricevitori
    power_combined = power_rx.mean(axis=-1)

    # --------------------------------------------------
    # 5. ASSI FISICI
    # --------------------------------------------------

    doppler_hz = np.fft.fftshift(
        np.fft.fftfreq(
            STFT_NFFT,
            d=CHIRP_PERIOD
        )
    )

    wavelength = 299792458 / 59e9

    velocity = doppler_hz * wavelength / 2

    # Tempo del centro di ogni finestra STFT
    centers = np.array([
        s + (STFT_WINDOW - 1) / 2
        for s in starts
    ])

    stft_time = (
        frame_time[:, None]
        + centers[None, :] * CHIRP_PERIOD
    )

    return (
        power_combined,
        power_rx,
        stft_complex,
        doppler_hz,
        velocity,
        stft_time
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        help="Percorso del file dreamhat_dataset.npz"
    )

    parser.add_argument(
        "--output",
        default=None
    )

    args = parser.parse_args()

    input_path = Path(args.dataset)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(
            input_path.stem + "_microdoppler.npz"
        )
    )

    # Caricamento dataset
    with np.load(input_path) as data:

        adc = data["adc"]

        if "frame_time_s" in data:
            frame_time = data["frame_time_s"]
        else:
            frame_time = (
                np.arange(adc.shape[0]) * FRAME_PERIOD
            )

        sequence = (
            data["sequence"]
            if "sequence" in data
            else np.arange(adc.shape[0])
        )

    print("ADC:", adc.shape)

    # Elaborazione
    (
        micro_doppler,
        micro_doppler_rx,
        stft_complex,
        doppler_hz,
        velocity,
        stft_time
    ) = process_adc(adc, frame_time)

    # Salvataggio
    np.savez_compressed(
        output_path,

        micro_doppler_power=micro_doppler,
        micro_doppler_power_per_rx=micro_doppler_rx,
        micro_doppler_complex_per_rx=stft_complex,

        doppler_hz=doppler_hz,
        velocity_m_s=velocity,

        frame_time_s=frame_time,
        stft_time_s=stft_time,
        sequence=sequence,

        chirp_period_s=CHIRP_PERIOD,
        frame_period_s=FRAME_PERIOD,

        stft_window=STFT_WINDOW,
        stft_hop=STFT_HOP,
        stft_nfft=STFT_NFFT
    )

    print("\nElaborazione completata!")
    print("Micro-Doppler:", micro_doppler.shape)
    print("Micro-Doppler per RX:", micro_doppler_rx.shape)
    print("STFT complessa:", stft_complex.shape)
    print("Salvato in:", output_path)


if __name__ == "__main__":
    main()
