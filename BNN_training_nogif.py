import csv
import numpy as np
import tensorflow as tf
import larq as lq
from pathlib import Path

# Configurazioni
THRESHOLD = 0.15
CATEGORIES = {"static": 0, "walking": 1, "bending_sit": 2, "fall": 3}
NUM_CLASSES = len(CATEGORIES)
WINDOW_SIZE = 10  # Stack di 10 frame (circa 1 secondo) per dare contesto temporale alla rete

def load_and_preprocess_data(csv_path, npz_folder):
    """Legge il CSV, carica i file .npz, applica soglia e crea finestre temporali."""
    # 1. Lettura annotazioni
    annotations = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            annotations.setdefault(row["file_npz"], []).append(
                (float(row["start_s"]), float(row["end_s"]), CATEGORIES[row["category"]])
            )

    X_list, y_list = [], []
    
    # 2. Elaborazione di ogni registrazione
    for npz_filename, events in annotations.items():
        data = np.load(Path(npz_folder) / npz_filename)
        rd_power = data["range_doppler_power"]
        times = data["frame_time_s"]
        
        # Normalizzazione Min-Max e Binarizzazione {-1, +1}
        rd_min = rd_power.min(axis=(1, 2), keepdims=True)
        rd_max = rd_power.max(axis=(1, 2), keepdims=True)
        rd_norm = (rd_power - rd_min) / (rd_max - rd_min + 1e-8)
        rd_bin = np.where(rd_norm >= THRESHOLD, 1.0, -1.0).astype(np.float32)
        
        # Etichettatura frame per frame
        y_frames = np.zeros(len(times), dtype=np.int32)
        for start_s, end_s, cat_id in events:
            mask = (times >= start_s) & (times < end_s)
            y_frames[mask] = cat_id
            
        # Creazione sliding windows (Early Fusion)
        for t in range(len(rd_bin) - WINDOW_SIZE):
            window = rd_bin[t : t + WINDOW_SIZE]           # Forma: (10, 64, 64)
            window = np.transpose(window, (1, 2, 0))       # Forma: (64, 64, 10 canali temporali)
            X_list.append(window)
            # L'etichetta della finestra è quella del frame finale (o centrale)
            y_list.append(y_frames[t + WINDOW_SIZE - 1])

    return np.array(X_list), np.array(y_list)

def build_bnn(input_shape=(64, 64, WINDOW_SIZE), num_classes=NUM_CLASSES):
    """Architettura BNN per Edge Computing."""
    kwargs = dict(input_quantizer="ste_sign", kernel_quantizer="ste_sign", kernel_constraint="weight_clip")
    
    model = tf.keras.Sequential([
        lq.layers.QuantConv2D(32, (3, 3), padding="same", input_shape=input_shape, **kwargs),
        tf.keras.layers.BatchNormalization(scale=False),
        tf.keras.layers.MaxPooling2D((2, 2)),

        lq.layers.QuantConv2D(64, (3, 3), padding="same", **kwargs),
        tf.keras.layers.BatchNormalization(scale=False),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        lq.layers.QuantDense(128, **kwargs),
        tf.keras.layers.BatchNormalization(scale=False),

        # Output Softmax standard per ottenere le probabilità
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

def train_pipeline():
    print("Caricamento dataset...")
    X, y = load_and_preprocess_data("labels.csv", "./dataset_folder")
    
    model = build_bnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("Avvio addestramento BNN...")
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
    
    # Salvataggio del modello pronto per il deployment
    model.save("fall_detector_bnn.h5")
    print("Modello salvato con successo.")

if __name__ == "__main__":
    train_pipeline()
