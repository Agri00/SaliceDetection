#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 2 DELLA PIPELINE: Caricamento dataset_npz + labels.csv -> Training BNN (Larq) -> fall_detector_bnn.tflite
"""

import csv
import random
from pathlib import Path

import larq as lq
import numpy as np
import tensorflow as tf

try:
    import larq_compute_engine as lce
    HAS_LCE = True
except ImportError:
    HAS_LCE = False

# ============================================================
# COSTANTI CONDIVISE CON PROCESSING E INFERENZA
# ============================================================
IMAGE_SIZE = 24
WINDOW_SIZE = 10
CATEGORIES = {"static": 0, "bending_sit": 1, "fall": 2}
NUM_CLASSES = len(CATEGORIES)
MODEL_FILENAME = "fall_detector_bnn.tflite"


def build_windows_from_recordings(rec_items, npz_folder: Path):
    """Carica X_bin (24x24) e time_s dai file .npz e costruisce finestre (24, 24, 10)."""
    X_list, y_list = [], []

    for npz_filename, events in rec_items:
        npz_path = npz_folder / npz_filename
        if not npz_path.is_file():
            print(f"[AVVISO] File mancante saltato: {npz_path}")
            continue

        data = np.load(npz_path)
        X_bin = data["X_bin"].astype(np.float32)  # (N_frames, 24, 24) già in {-1.0, +1.0}
        time_s = data["time_s"]

        if len(X_bin) < WINDOW_SIZE:
            continue

        # Assegnazione etichette frame per frame (default = 0: static)
        y_frames = np.zeros(len(time_s), dtype=np.int32)
        for start_s, end_s, cat_id in events:
            mask = (time_s >= start_s) & (time_s < end_s)
            y_frames[mask] = cat_id

        # Impilamento temporale (Early Fusion): (24, 24, WINDOW_SIZE)
        for t in range(len(X_bin) - WINDOW_SIZE + 1):
            window = np.transpose(X_bin[t : t + WINDOW_SIZE], (1, 2, 0))
            X_list.append(window)
            y_list.append(y_frames[t + WINDOW_SIZE - 1])

    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.int32)


def load_dataset_recording_split(csv_path: str, npz_folder: str, val_ratio: float = 0.2, seed: int = 42):
    """Divide le registrazioni tra training e validation senza mescolare frame dello stesso evento."""
    annotations = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat_name = row["category"].strip()
            if cat_name not in CATEGORIES:
                raise ValueError(f"Categoria '{cat_name}' non valida. Usa: {list(CATEGORIES.keys())}")
            annotations.setdefault(row["file_npz"].strip(), []).append(
                (float(row["start_s"]), float(row["end_s"]), CATEGORIES[cat_name])
            )

    # Raggruppa per categoria principale della registrazione per uno split stratificato
    by_main_cat = {cat_id: [] for cat_id in CATEGORIES.values()}
    for item in annotations.items():
        _, events = item
        main_cat = max((e[2] for e in events), default=0)
        by_main_cat[main_cat].append(item)

    rng = random.Random(seed)
    train_recs, val_recs = [], []
    for cat_id, items in by_main_cat.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
        val_recs.extend(items[:n_val])
        train_recs.extend(items[n_val:])

    folder = Path(npz_folder)
    X_train, y_train = build_windows_from_recordings(train_recs, folder)
    X_val, y_val = build_windows_from_recordings(val_recs, folder)
    return (X_train, y_train), (X_val, y_val)


def compute_balanced_weights(y_train: np.ndarray) -> dict:
    """Calcola pesi inversamente proporzionali alla frequenza delle 3 classi."""
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    total = len(y_train)
    weights = {
        cls_id: float(total / (NUM_CLASSES * max(count, 1)))
        for cls_id, count in enumerate(counts)
    }
    print(f"Finestre training per classe {dict(zip(CATEGORIES.keys(), counts))}")
    print(f"Class weights applicati: {weights}")
    return weights


def build_bnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, WINDOW_SIZE), num_classes=NUM_CLASSES):
    """Architettura BNN 2D per input (24, 24, 10) e 3 macro-categorie."""
    kwargs = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
    )

    return tf.keras.Sequential([
        lq.layers.QuantConv2D(32, (3, 3), padding="same", input_shape=input_shape, **kwargs),
        tf.keras.layers.BatchNormalization(scale=False),
        tf.keras.layers.MaxPooling2D((2, 2)),  # -> 12x12

        lq.layers.QuantConv2D(64, (3, 3), padding="same", **kwargs),
        tf.keras.layers.BatchNormalization(scale=False),
        tf.keras.layers.MaxPooling2D((2, 2)),  # -> 6x6

        lq.layers.QuantConv2D(128, (3, 3), padding="same", **kwargs),
        tf.keras.layers.BatchNormalization(scale=False),
        tf.keras.layers.MaxPooling2D((2, 2)),  # -> 3x3

        tf.keras.layers.Flatten(),
        lq.layers.QuantDense(128, **kwargs),
        tf.keras.layers.BatchNormalization(scale=False),

        # Strato finale in precisione standard: restituisce [p_static, p_bending_sit, p_fall]
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def export_tflite(model: tf.keras.Model, output_path: Path):
    """Esporta il modello in .tflite (impacchettato a 1-bit se LCE è disponibile)."""
    if HAS_LCE:
        tflite_bytes = lce.convert_keras_model(model)
        mode = "Larq Compute Engine (1-bit packed)"
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_bytes = converter.convert()
        mode = "TFLite standard"

    output_path.write_bytes(tflite_bytes)
    print(f"Modello salvato in '{output_path}' ({output_path.stat().st_size / 1024:.1f} KB) [{mode}]")


def main():
    (X_train, y_train), (X_val, y_val) = load_dataset_recording_split(
        csv_path="labels.csv",
        npz_folder="./dataset_npz",
        val_ratio=0.2,
    )

    class_weights = compute_balanced_weights(y_train)
    model = build_bnn()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005, decay_steps=500, decay_rate=0.9
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        epochs=100,
        batch_size=32,
    )

    model.save("fall_detector_bnn.h5")
    export_tflite(model, Path(MODEL_FILENAME))


if __name__ == "__main__":
    main()
