
import numpy as np
from PIL import Image
from pathlib import Path

# ==========================================
# 1. SELEZIONE FILE
# ==========================================

file_path = input(
    "Inserisci il percorso del file .npz da elaborare: "
).strip().strip('"')

data = np.load(file_path)

print("File caricato correttamente!")

power = data["micro_doppler_power"]

print("Shape originale:", power.shape)

# ==========================================
# 2. PARAMETRI
# ==========================================

threshold = 0.08
target_size = (24, 24)

X_gray = []
X_bin = []

frame_index = []
window_index = []
time_index = []

# ==========================================
# 3. ELABORAZIONE
# ==========================================

for frame in range(power.shape[0]):

    for window in range(power.shape[1]):

        P = power[frame, window]

        amplitude = np.sqrt(np.maximum(P, 0))

        p_min = amplitude.min()
        p_max = amplitude.max()

        if p_max > p_min:
            normalized = (amplitude - p_min) / (p_max - p_min)
        else:
            normalized = np.zeros_like(amplitude)

        image_8bit = (normalized * 255).astype(np.uint8)

        image = Image.fromarray(image_8bit)

        image = image.resize(
            target_size,
            Image.Resampling.BILINEAR
        )

        resized = np.asarray(image).astype(np.float32) / 255.0

        binary = (resized >= threshold).astype(np.uint8)

        X_gray.append(resized)
        X_bin.append(binary)

        frame_index.append(frame)
        window_index.append(window)

        time_index.append(data["stft_time_s"][frame, window])

# ==========================================
# 4. CONVERSIONE ARRAY
# ==========================================

X_gray = np.asarray(X_gray, dtype=np.float32)
X_bin = np.asarray(X_bin, dtype=np.uint8)

frame_index = np.asarray(frame_index)
window_index = np.asarray(window_index)
time_index = np.asarray(time_index)

labels = np.full(len(X_bin), -1, dtype=np.int8)

# ==========================================
# 5. SALVATAGGIO
# ==========================================

input_path = Path(file_path)

output_path = input_path.with_name(
    input_path.stem + "_binary_dataset.npz"
)

np.savez_compressed(
    output_path,
    X_bin=X_bin,
    X_gray=X_gray,
    labels=labels,
    frame_index=frame_index,
    window_index=window_index,
    time_s=time_index,
    threshold=threshold
)

print("\nDataset creato!")
print("Numero immagini:", len(X_bin))
print("Shape binaria:", X_bin.shape)
print("Shape grayscale:", X_gray.shape)
print("Valori binari:", np.unique(X_bin))
print("Salvato in:", output_path)
