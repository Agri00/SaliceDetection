#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 3 DELLA PIPELINE: Inferenza Real-Time su Raspberry Pi 3 Model B (aarch64)
Carica 'fall_detector_bnn.tflite' e replica esattamente il pre-processing di 1_process_recordings.py.
"""

import struct
import time
from collections import deque

import numpy as np
from PIL import Image

try:
    import larq_compute_engine as lce
    BACKEND = "LCE"
except ImportError:
    import tflite_runtime.interpreter as tflite
    BACKEND = "TFLITE"

# ============================================================
# COSTANTI CONDIVISE CON PROCESSING E TRAINING
# ============================================================
N_CHIRPS = 64
N_SAMPLES = 128
N_RX = 3

HEADER_BYTES = 12
EXPECTED_SAMPLES = N_CHIRPS * N_SAMPLES * N_RX
EXPECTED_PAYLOAD_BYTES = (EXPECTED_SAMPLES // 2) * 3  # 36.864 byte

IMAGE_SIZE = 24
THRESHOLD = 0.25
NORM_WINDOW = 10
MIN_DB_SPAN = 15.0
WINDOW_SIZE = 10

CATEGORIES = {"static": 0, "bending_sit": 1, "fall": 2}
MODEL_FILENAME = "fall_detector_bnn.tflite"

WIN_RANGE = np.hanning(N_SAMPLES).astype(np.float32)[None, :, None]
WIN_DOPPLER = np.hanning(N_CHIRPS).astype(np.float32)[:, None, None]


def unpack_12bit_fifo(payload: bytes) -> np.ndarray:
    """Converte word FIFO a 24-bit in due campioni ADC a 12-bit."""
    b = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.uint16)
    sample0 = (b[:, 0] << 4) | (b[:, 1] >> 4)
    sample1 = ((b[:, 1] & 0x0F) << 8) | b[:, 2]
    out = np.empty(sample0.size * 2, dtype=np.uint16)
    out[0::2] = sample0
    out[1::2] = sample1
    return out


def compute_single_frame_rd_db(raw_payload: bytes) -> np.ndarray:
    """Calcola la mappa Range-Doppler in dB di un singolo frame -> (64, 64)."""
    samples = unpack_12bit_fifo(raw_payload)
    x = samples.reshape(N_CHIRPS, N_SAMPLES, N_RX).astype(np.float32)

    # 1. Rimozione componente continua (offset ADC) lungo fast-time
    x -= x.mean(axis=1, keepdims=True)

    # 2. Range FFT -> metà positiva (64 bin)
    x *= WIN_RANGE
    range_fft = np.fft.fft(x, n=N_SAMPLES, axis=1)[:, :64, :].astype(np.complex64)

    # 3. Clutter removal (allineato a compute_rd_power_db in 1_process_recordings.py)
    range_fft -= range_fft.mean(axis=0, keepdims=True)

    # 4. Doppler FFT sui 64 chirp
    range_fft *= WIN_DOPPLER
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, n=N_CHIRPS, axis=0), axes=0)
    doppler_fft = np.transpose(doppler_fft, (1, 0, 2))  # (Range=64, Doppler=64, RX=3)

    power_mean = (np.abs(doppler_fft) ** 2).mean(axis=-1).astype(np.float32)
    return 10.0 * np.log10(np.maximum(power_mean, 1e-12))


class BNNInterpreter:
    """Wrapper unificato per LCE o TFLite Runtime su Raspberry Pi OS aarch64."""

    def __init__(self, tflite_path: str):
        self.backend = BACKEND
        if self.backend == "LCE":
            with open(tflite_path, "rb") as f:
                self.interpreter = lce.testing.Interpreter(f.read())
        else:
            self.interpreter = tflite.Interpreter(model_path=tflite_path, num_threads=2)
            self.interpreter.allocate_tensors()
            self.input_idx = self.interpreter.get_input_details()[0]["index"]
            self.output_idx = self.interpreter.get_output_details()[0]["index"]

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.backend == "LCE":
            return self.interpreter.predict(input_tensor)[0]
        self.interpreter.set_tensor(self.input_idx, input_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_idx)[0]


class EdgeFallDetector:
    def __init__(self, tflite_path: str = MODEL_FILENAME, cooldown_frames: int = 30):
        self.model = BNNInterpreter(tflite_path)

        # Storico dB per normalizzazione causale identica a normalize_resize_binarize
        self.db_history = deque(maxlen=NORM_WINDOW)

        # Tensore pre-allocato (1, 24, 24, 10) in {-1.0, +1.0}
        self.input_tensor = np.full((1, IMAGE_SIZE, IMAGE_SIZE, WINDOW_SIZE), -1.0, dtype=np.float32)
        self.frames_collected = 0
        self.last_seq = None

        # Storico punteggi Softmax (1.5 secondi a 10 Hz)
        self.score_history = deque(maxlen=15)
        self.cooldown_frames = cooldown_frames
        self.cooldown_counter = 0

    def _preprocess_frame(self, power_db: np.ndarray) -> np.ndarray:
        """Replica esatta di normalize_resize_binarize() su singolo frame live."""
        self.db_history.append(power_db)
        history_arr = np.asarray(self.db_history, dtype=np.float32)

        db_min = float(np.percentile(history_arr, 10))
        db_max = max(float(np.percentile(history_arr, 99.5)), db_min + MIN_DB_SPAN)

        gray = np.clip((power_db - db_min) / (db_max - db_min + 1e-12), 0.0, 1.0)
        gray_uint8 = np.round(gray * 255.0).astype(np.uint8)

        pil_img = Image.fromarray(gray_uint8).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        x_gray = np.asarray(pil_img, dtype=np.float32) / 255.0
        return np.where(x_gray >= THRESHOLD, 1.0, -1.0).astype(np.float32)

    def push_raw_packet(self, header_bytes: bytes, payload_bytes: bytes) -> dict:
        """Elabora un frame SPI del DreamHAT+ e valuta il contesto temporale."""
        version, seq, payload_len = struct.unpack("<III", header_bytes)

        dropped = 0
        if self.last_seq is not None and seq != self.last_seq + 1:
            dropped = seq - (self.last_seq + 1)
        self.last_seq = seq

        if version != 0 or payload_len != EXPECTED_PAYLOAD_BYTES:
            return {"status": "ERRORE_PACKET", "dropped_frames": dropped}

        power_db = compute_single_frame_rd_db(payload_bytes)
        bin_frame = self._preprocess_frame(power_db)

        # Scorrimento in-place del buffer circolare (24x24x10)
        self.input_tensor[0, :, :, :-1] = self.input_tensor[0, :, :, 1:]
        self.input_tensor[0, :, :, -1] = bin_frame
        self.frames_collected += 1

        if self.frames_collected < WINDOW_SIZE:
            return {"status": "INIT_BUFFER", "dropped_frames": dropped}

        scores = self.model.predict(self.input_tensor)
        self.score_history.append(scores)
        return self._evaluate_event(scores, dropped)

    def _evaluate_event(self, current_scores: np.ndarray, dropped: int) -> dict:
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        if len(self.score_history) < 15:
            return {"status": "MONITORING", "scores": current_scores, "dropped_frames": dropped}

        history = np.asarray(self.score_history)
        fall_probs = history[:, CATEGORIES["fall"]]
        static_probs = history[:, CATEGORIES["static"]]

        # Media mobile su 0.5s + verifica immobilità post-impatto negli ultimi 3 frame
        smoothed_fall = float(np.mean(fall_probs[-5:]))
        post_static = float(np.mean(static_probs[-3:]))

        reliability_score = smoothed_fall * (0.6 + 0.4 * post_static)
        fall_detected = False

        if reliability_score >= 0.70 and self.cooldown_counter == 0:
            fall_detected = True
            self.cooldown_counter = self.cooldown_frames

        return {
            "status": "FALL_DETECTED" if fall_detected else "OK",
            "fall_detected": fall_detected,
            "reliability_score": round(reliability_score, 3),
            "scores": np.round(current_scores, 3).tolist(),  # [static, bending_sit, fall]
            "dropped_frames": dropped,
        }


def trigger_alarm(event_info: dict):
    print(f"[{time.strftime('%H:%M:%S')}] ⚠️ ALLARME CADUTA! Score={event_info['reliability_score']}")


if __name__ == "__main__":
    detector = EdgeFallDetector(MODEL_FILENAME)
    print(f"Backend attivo: {detector.model.backend}")

    # Test su registrazione .bin (sostituibile con lo stream live di utility.BGT60TR13C)
    with open("test_caduta.bin", "rb") as stream:
        while True:
            header = stream.read(HEADER_BYTES)
            if len(header) < HEADER_BYTES:
                break
            payload = stream.read(EXPECTED_PAYLOAD_BYTES)
            if len(payload) < EXPECTED_PAYLOAD_BYTES:
                break

            t0 = time.perf_counter()
            res = detector.push_raw_packet(header, payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if res.get("fall_detected"):
                trigger_alarm(res)
            elif res["status"] == "OK":
                print(f"{elapsed_ms:.1f} ms | Score={res['reliability_score']} | P={res['scores']}")
