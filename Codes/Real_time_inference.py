
"""
Inferenza Real-Time su Raspberry Pi 3 Model B (aarch64)
Carica 'bnn_fall_detection.pt' e 'noise_reference.npz', replicando esattamente
il pre-processing di radar_common.py, ProcessingTutteRegistrazioni.py e BNN_training_aggiornato.py.
"""

import argparse
import contextlib
import io
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

# Forza Python a cercare i moduli nella stessa cartella di questo script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from radar_common import (
    ADC_SIGNED,
    EXPECTED_PAYLOAD_SIZE,
    HEADER_SIZE,
    N_CHIRPS,
    N_RX,
    N_SAMPLES,
    STFT_NFFT,
    calculate_microdoppler,
)
from BNN_training_aggiornato import (
    RadarFallBNN,
    WIN_STFT_STEPS,
    format_binary_window_for_bnn,
)

# ============================================================
# COSTANTI ALLINEATE CON ProcessingTutteRegistrazioni.py
# ============================================================

NOISE_MARGIN_DB = 6          # Identico a ProcessingTutteRegistrazioni.py
USE_ADAPTIVE_NOISE = True    # Identico al 60° percentile locale usato nel processing
USE_MORPH_FILTER = True      # Identico al filtro morfologico 3x3 (neighbors >= 3)

# ============================================================
# PARAMETRI DI ALLARME REAL-TIME
# ============================================================

PROB_THRESHOLD = 0.50        # Allinea alla soglia migliore stampata sul Test Set
CONFIRMATION_FRAMES = 3      # Frame consecutivi sopra soglia (~0.3 s) per confermare la caduta
COOLDOWN_FRAMES = 50         # Pausa di 5.0 s (a 10 Hz) dopo un allarme confermato

MODEL_FILENAME = "bnn_fall_detection.pt"
NOISE_REF_FILENAME = "noise_reference.npz"


def unpack_12bit_fifo(payload: bytes) -> np.ndarray:
    """Converte word FIFO a 24-bit in due campioni ADC a 12-bit centrati."""
    b = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.uint16)
    sample0 = (b[:, 0] << 4) | (b[:, 1] >> 4)
    sample1 = ((b[:, 1] & 0x0F) << 8) | b[:, 2]

    out = np.empty(sample0.size * 2, dtype=np.uint16)
    out[0::2] = sample0
    out[1::2] = sample1

    if ADC_SIGNED:
        adc = out.astype(np.int32)
        adc[adc >= 2048] -= 4096
        return adc.astype(np.float32)

    # Campioni unsigned offset-binary a 12-bit centrati attorno a metà scala (2048)
    return out.astype(np.float32) - 2048.0


class EdgeFallDetector:
    def __init__(
        self,
        weights_path: str | Path = MODEL_FILENAME,
        noise_ref_path: str | Path = NOISE_REF_FILENAME,
        cooldown_frames: int = COOLDOWN_FRAMES,
    ):
        # Ottimizzazione CPU quad-core ARM Cortex-A53 del Raspberry Pi 3B
        torch.set_num_threads(4)
        self.device = torch.device("cpu")

        # 1. Carica il profilo di rumore di stanza vuota (noise_reference.npz)
        ref_data = np.load(noise_ref_path)
        self.noise_reference = ref_data["noise_reference"].astype(np.float64)
        if self.noise_reference.shape != (STFT_NFFT,):
            raise ValueError(
                f"noise_reference deve avere shape ({STFT_NFFT},), trovato {self.noise_reference.shape}"
            )

        self.margin_linear = 10.0 ** (NOISE_MARGIN_DB / 10.0)
        self.base_threshold_power = self.noise_reference * self.margin_linear

        # 2. Carica i pesi della BNN PyTorch addestrata (.pt)
        self.model = RadarFallBNN().to(self.device)
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 3. Buffer circolari a memoria fissa
        self.stft_buffer = deque(maxlen=WIN_STFT_STEPS)  # 48 righe STFT = 16 frame = 1.6 s
        self.power_history = deque(maxlen=210)           # Ultimi ~7 s (210 step STFT) per rumore locale

        self.last_seq = None
        self.consecutive_positives = 0
        self.cooldown_frames = cooldown_frames
        self.cooldown_counter = 0

    def push_raw_packet(self, header_bytes: bytes, payload_bytes: bytes) -> dict:
        """Elabora un singolo pacchetto SPI del DreamHAT+ e valuta il contesto temporale."""
        version, seq, payload_len = struct.unpack("<III", header_bytes)

        dropped = 0
        if self.last_seq is not None and seq != self.last_seq + 1:
            dropped = seq - (self.last_seq + 1)
        self.last_seq = seq

        if version != 0 or payload_len != EXPECTED_PAYLOAD_SIZE:
            return {"status": "ERRORE_PACKET", "dropped_frames": dropped}

        # Decodifica frame ADC (64, 128, 3)
        samples = unpack_12bit_fifo(payload_bytes)
        adc_frame = samples.reshape(N_CHIRPS, N_SAMPLES, N_RX)

        # Calcolo micro-Doppler tramite radar_common.py (ROI range + Top-K + Clutter removal)
        with contextlib.redirect_stdout(io.StringIO()):
            microdoppler_power, _, _ = calculate_microdoppler(
                adc_frame[None, ...],
                np.array([seq], dtype=np.uint32),
            )

        # Soglia adattiva identica a ProcessingTutteRegistrazioni.py
        if USE_ADAPTIVE_NOISE:
            for p_row in microdoppler_power:
                self.power_history.append(p_row)
            local_noise = np.percentile(np.asarray(self.power_history), 60, axis=0)
            effective_noise = np.maximum(self.noise_reference, local_noise)
            threshold_power = effective_noise * self.margin_linear
        else:
            threshold_power = self.base_threshold_power

        # Binarizzazione e inserimento nel buffer temporale
        binary_rows = (microdoppler_power > threshold_power[None, :]).astype(np.uint8)
        for row in binary_rows:
            self.stft_buffer.append(row)

        if len(self.stft_buffer) < WIN_STFT_STEPS:
            return {"status": "INIT_BUFFER", "dropped_frames": dropped}

        window_2d = np.asarray(self.stft_buffer, dtype=np.uint8)  # shape (48, 64)

        # Filtro morfologico 3x3 (rimuove pixel isolati, mantiene cluster >= 3 pixel)
        if USE_MORPH_FILTER:
            padded = np.pad(window_2d, ((1, 1), (1, 1)), mode="constant")
            neighbors = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            )
            window_2d = ((window_2d == 1) & (neighbors >= 3)).astype(np.uint8)

        # Ridimensionamento a 24x24 in {-1.0, +1.0} e inferenza BNN
        bnn_input = format_binary_window_for_bnn(window_2d)
        tensor_x = torch.from_numpy(bnn_input).unsqueeze(0).unsqueeze(0)  # (1, 1, 24, 24)

        with torch.inference_mode():
            fall_prob = float(torch.sigmoid(self.model(tensor_x)).item())

        return self._evaluate_event(fall_prob, dropped)

    def _evaluate_event(self, fall_prob: float, dropped: int) -> dict:
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        if fall_prob >= PROB_THRESHOLD:
            self.consecutive_positives += 1
        else:
            self.consecutive_positives = 0

        fall_detected = False
        if (
            self.consecutive_positives >= CONFIRMATION_FRAMES
            and self.cooldown_counter == 0
        ):
            fall_detected = True
            self.cooldown_counter = self.cooldown_frames

        return {
            "status": "FALL_DETECTED" if fall_detected else "OK",
            "fall_detected": fall_detected,
            "fall_prob": round(fall_prob, 3),
            "consecutive_hits": self.consecutive_positives,
            "dropped_frames": dropped,
        }


def trigger_alarm(event_info: dict):
    print(
        f"[{time.strftime('%H:%M:%S')}] ALLARME CADUTA! "
        f"Prob={event_info['fall_prob']} (Consecutivi={event_info['consecutive_hits']})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferenza Real-Time BNN Fall Detection")
    parser.add_argument("--weights", type=Path, default=Path(SCRIPT_DIR) / MODEL_FILENAME)
    parser.add_argument("--noise-ref", type=Path, default=Path(SCRIPT_DIR) / NOISE_REF_FILENAME)
    parser.add_argument("--bin-file", type=Path, help="Percorso del file .bin di test")
    parser.add_argument("--realtime", action="store_true", help="Simula cadenza reale a 10 Hz")
    args = parser.parse_args()

    bin_path = args.bin_file
    if bin_path is None:
        default_test = Path(SCRIPT_DIR) / "test_caduta.bin"
        if default_test.exists():
            bin_path = default_test
        else:
            from tkinter import Tk, filedialog
            root = Tk()
            root.withdraw()
            picked = filedialog.askopenfilename(
                title="Seleziona un file .bin di test",
                filetypes=[("BIN files", "*.bin")],
            )
            root.destroy()
            if not picked:
                sys.exit(0)
            bin_path = Path(picked)

    detector = EdgeFallDetector(args.weights, args.noise_ref)
    print(f"Modello PyTorch caricato | Avvio elaborazione su: {bin_path.name}")

    # Test su registrazione .bin (sostituibile con lo stream live di utility.BGT60TR13C)
    with open(bin_path, "rb") as stream:
        while True:
            header = stream.read(HEADER_SIZE)
            if len(header) < HEADER_SIZE:
                break
            payload = stream.read(EXPECTED_PAYLOAD_SIZE)
            if len(payload) < EXPECTED_PAYLOAD_SIZE:
                break

            t0 = time.perf_counter()
            res = detector.push_raw_packet(header, payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if res.get("fall_detected"):
                trigger_alarm(res)
            elif res["status"] == "OK":
                print(
                    f"{elapsed_ms:.1f} ms | Prob={res['fall_prob']:.3f} | "
                    f"Hits={res['consecutive_hits']}/{CONFIRMATION_FRAMES}"
                )

            if args.realtime:
                time.sleep(max(0.0, 0.1 - (elapsed_ms / 1000.0)))
