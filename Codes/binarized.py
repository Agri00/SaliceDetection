
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# ==========================================
# 1. SELEZIONE DATASET
# ==========================================

print("\n=== GENERAZIONE GIF DATASET ===")

file_path = input(
    "\nInserisci il percorso completo del dataset .npz:\n> "
).strip().strip('"')

if not Path(file_path).is_file():
    print("\nERRORE: file non trovato!")
    input("Premi INVIO per uscire...")
    raise SystemExit

print("\nCaricamento dataset...")

data = np.load(file_path)

X_bin = data["X_bin"]
X_gray = data["X_gray"]

frames = data["frame_index"]
windows = data["window_index"]
times = data["time_s"]

print("Dataset caricato correttamente!")
print("Numero immagini:", len(X_bin))
print("Dimensioni:", X_bin.shape)

# ==========================================
# 2. ORDINAMENTO TEMPORALE
# ==========================================

order = np.lexsort((windows, frames))

X_bin = X_bin[order]
X_gray = X_gray[order]

frames = frames[order]
windows = windows[order]
times = times[order]

# ==========================================
# 3. PARAMETRI GIF
# ==========================================

fps = 10

output_dir = Path(file_path).parent

# ==========================================
# 4. FUNZIONE GENERAZIONE GIF
# ==========================================

def create_gif(images, title, filename):

    print("\nGenerazione:", filename)

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        images[0],
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):

        im.set_data(images[i])

        ax.set_title(
            f"{title}\n"
            f"Frame {frames[i]} - "
            f"Finestra {windows[i]+1}\n"
            f"Tempo: {times[i]:.3f} s"
        )

        return [im]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=1000/fps,
        blit=False
    )

    animation.save(
        str(filename),
        writer=PillowWriter(fps=fps)
    )

    plt.close(fig)

    print("Completata:", filename)

# ==========================================
# 5. GENERAZIONE GIF GRAYSCALE
# ==========================================

create_gif(
    X_gray,
    "Dataset grayscale",
    output_dir / "dataset_grayscale.gif"
)

# ==========================================
# 6. GENERAZIONE GIF BINARIA
# ==========================================

create_gif(
    X_bin,
    "Dataset binario",
    output_dir / "dataset_binary.gif"
)

# ==========================================
# 7. FINE
# ==========================================

print("\n================================")
print("GENERAZIONE COMPLETATA!")
print("================================")

print("\nFile salvati in:")
print(output_dir)

input("\nPremi INVIO per chiudere...")
