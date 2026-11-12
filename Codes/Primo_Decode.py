#!/usr/bin/env python3
"""
Decode DreamHAT+ BGT60TR13C recordings produced by utility.BGT60TR13C.

PER CHIAMARE LA FUNZIONE:
py dreamhat_decode.py NOME_FILE.bin

oppure se il file é in una directory diversa:
py dreamhat_decode.py "DIRECTORY"


File layout (as written by the supplied driver):
    uint32_le version
    uint32_le sequence_number
    uint32_le payload_length
    payload_length bytes of packed 12-bit ADC samples

For the supplied 3-RX / 64-chirp / 128-sample configuration:
    ADC stream order = ADC1, ADC2, ADC3, ADC1, ADC2, ADC3, ...
    one frame = (64 chirps, 128 samples/chirp, 3 RX) = 24,576 samples.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

# Acquisition parameters from the supplied settings JSON
N_CHIRPS = 64
N_SAMPLES_PER_CHIRP = 128
N_RX = 3
FS_HZ = 2_000_000.0
CHIRP_PERIOD_S = 0.0010007125092670321
FRAME_PERIOD_S = 0.10000739991664886
F_START_HZ = 58e9
F_END_HZ = 60e9
C_M_S = 299_792_458.0
FC_HZ = 0.5 * (F_START_HZ + F_END_HZ)

HEADER_BYTES = 12
EXPECTED_SAMPLES = N_CHIRPS * N_SAMPLES_PER_CHIRP * N_RX
EXPECTED_PAYLOAD_BYTES = (EXPECTED_SAMPLES // 2) * 3
EXPECTED_FRAME_BYTES = HEADER_BYTES + EXPECTED_PAYLOAD_BYTES


def unpack_12bit_fifo(payload: bytes) -> np.ndarray:
    """Unpack each 24-bit FIFO word into two 12-bit ADC values.

    FIFO word bits 23:12 are Data 0; bits 11:0 are Data 1.
    The SPI byte stream is most-significant byte first.
    """
    if len(payload) % 3:
        raise ValueError(f"Payload length {len(payload)} is not divisible by 3.")

    b = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.uint16)
    sample0 = (b[:, 0] << 4) | (b[:, 1] >> 4)
    sample1 = ((b[:, 1] & 0x0F) << 8) | b[:, 2]

    out = np.empty(sample0.size * 2, dtype=np.uint16)
    out[0::2] = sample0
    out[1::2] = sample1
    return out


def read_recording(path: Path):
    raw = path.read_bytes()
    offset = 0
    frames = []
    sequence_numbers = []

    while offset < len(raw):
        if len(raw) - offset < HEADER_BYTES:
            raise ValueError(f"Incomplete frame header at byte offset {offset}.")

        version, seq, payload_len = struct.unpack_from("<III", raw, offset)
        if version != 0:
            raise ValueError(
                f"Unexpected format version {version} at byte offset {offset}."
            )
        if payload_len != EXPECTED_PAYLOAD_BYTES:
            raise ValueError(
                f"Frame {seq}: payload length is {payload_len}, "
                f"expected {EXPECTED_PAYLOAD_BYTES}. Check radar settings."
            )

        start = offset + HEADER_BYTES
        end = start + payload_len
        if end > len(raw):
            raise ValueError(f"Frame {seq} is truncated.")

        samples = unpack_12bit_fifo(raw[start:end])
        if samples.size != EXPECTED_SAMPLES:
            raise ValueError(
                f"Frame {seq}: decoded {samples.size} samples, "
                f"expected {EXPECTED_SAMPLES}."
            )

        # With three active ADCs, the BGT60TR13C FIFO cycles ADC1, ADC2,
        # ADC3 for each sampling trigger. Preserve RX channels as a final axis.
        frame = samples.reshape(N_CHIRPS, N_SAMPLES_PER_CHIRP, N_RX)
        frames.append(frame)
        sequence_numbers.append(seq)
        offset = end

    if not frames:
        raise ValueError("No complete frames found.")

    seq = np.asarray(sequence_numbers, dtype=np.uint32)
    expected_seq = np.arange(seq[0], seq[0] + len(seq), dtype=np.uint32)
    if not np.array_equal(seq, expected_seq):
        raise ValueError("Sequence numbers are not consecutive; check for dropped frames.")

    adc_cube = np.stack(frames, axis=0)  # (frame, chirp, fast_time_sample, RX)
    return adc_cube, seq


def make_range_doppler(adc_cube: np.ndarray):
    """Compute per-RX and RX-averaged Range-Doppler power maps.

    Range is kept on the positive beat-frequency half-spectrum.
    The distance calibration assumes the sampled chirp spans the configured
    58--60 GHz sweep over the 128 ADC samples; verify against register config.
    """
    n_range_fft = N_SAMPLES_PER_CHIRP
    n_doppler_fft = N_CHIRPS

    x = adc_cube.astype(np.float32)

    # Remove each chirp/RX fast-time mean (ADC offset / near-DC component).
    x -= x.mean(axis=2, keepdims=True)

    win_range = np.hanning(N_SAMPLES_PER_CHIRP).astype(np.float32)
    x *= win_range[None, None, :, None]

    # Range FFT: (frame, chirp, range_bin, RX)
    range_fft = np.fft.fft(x, n=n_range_fft, axis=2)
    range_fft = range_fft[:, :, : n_range_fft // 2, :]

    # Doppler FFT across the 64 slow-time chirps.
    win_doppler = np.hanning(N_CHIRPS).astype(np.float32)
    range_fft *= win_doppler[None, :, None, None]
    rd_complex = np.fft.fftshift(
        np.fft.fft(range_fft, n=n_doppler_fft, axis=1), axes=1
    )  # (frame, doppler_bin, range_bin, RX)

    power_per_rx = np.abs(rd_complex) ** 2
    power_per_rx = np.transpose(power_per_rx, (0, 3, 2, 1)).astype(np.float32)
    # (frame, RX, range_bin, doppler_bin)

    # Non-coherent combination: average powers, not complex voltages.
    power_combined = power_per_rx.mean(axis=1)  # (frame, range_bin, doppler_bin)

    doppler_hz = np.fft.fftshift(
        np.fft.fftfreq(n_doppler_fft, d=CHIRP_PERIOD_S)
    )
    wavelength_m = C_M_S / FC_HZ
    velocity_m_s = doppler_hz * wavelength_m / 2.0

    # Provisional calibration: assumes 2 GHz sweep over the 128-sample
    # acquisition interval. The register configuration should confirm this.
    bandwidth_hz = F_END_HZ - F_START_HZ
    range_resolution_m = C_M_S / (2.0 * bandwidth_hz)
    range_m = np.arange(n_range_fft // 2) * range_resolution_m

    return power_per_rx, power_combined, range_m, velocity_m_s


def main():
    parser = argparse.ArgumentParser(
        description="Decode DreamHAT+ BGT60TR13C .bin and create Range-Doppler arrays."
    )
    parser.add_argument("bin_file", type=Path, help="Input .bin recording")
    parser.add_argument(
        "--output", type=Path, default=Path("dreamhat_dataset.npz"),
        help="Output NumPy archive (default: dreamhat_dataset.npz)"
    )
    parser.add_argument(
        "--preview", type=Path, default=Path("range_doppler_preview.png"),
        help="Output PNG preview (default: range_doppler_preview.png)"
    )
    args = parser.parse_args()

    adc_cube, seq = read_recording(args.bin_file)
    rd_per_rx, rd_combined, range_m, velocity_m_s = make_range_doppler(adc_cube)

    frame_time_s = (seq - seq[0]).astype(np.float64) * FRAME_PERIOD_S

    np.savez_compressed(
        args.output,
        adc=adc_cube,                         # uint16, (frame, chirp, sample, RX)
        range_doppler_power=rd_combined,      # float32, (frame, range, Doppler)
        range_doppler_power_per_rx=rd_per_rx, # float32, (frame, RX, range, Doppler)
        sequence=seq,
        frame_time_s=frame_time_s,
        range_m=range_m,                      # provisional calibration
        velocity_m_s=velocity_m_s,
    )

    # Save a visual sanity-check of the first frame.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_db = 10.0 * np.log10(rd_combined[0] + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        image_db,
        origin="lower",
        aspect="auto",
        extent=[
            velocity_m_s[0], velocity_m_s[-1],
            range_m[0], range_m[-1],
        ],
    )
    ax.set_xlabel("Radial velocity (m/s)")
    ax.set_ylabel("Range (m, provisional calibration)")
    ax.set_title("DreamHAT+ Range-Doppler — frame 0")
    fig.colorbar(im, ax=ax, label="Power (dB, arbitrary scale)")
    fig.tight_layout()
    fig.savefig(args.preview, dpi=160)
    plt.close(fig)

    print(f"Decoded frames: {len(seq)}")
    print(f"ADC cube shape: {adc_cube.shape}")
    print(f"Range-Doppler shape: {rd_combined.shape}")
    print(f"Per-RX Range-Doppler shape: {rd_per_rx.shape}")
    print(f"Saved dataset: {args.output.resolve()}")
    print(f"Saved preview: {args.preview.resolve()}")
    print("NOTE: range_m is provisional until the register configuration is checked.")


if __name__ == "__main__":
    main()
