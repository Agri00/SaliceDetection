"""
Crea una GIF animata dalle mappe Range-Doppler contenute in dreamhat_dataset.npz.


PER CHIAMARE:
py crea_gif.py   per tutte le antenne combinate (file nella stessa cartella)

oppure se il file è in una directory diversa:
py crea_gif.py "DIRECTORY"

Uso:
    py crea_gif.py
    py crea_gif.py percorso/dreamhat_dataset.npz
    py crea_gif.py percorso/dreamhat_dataset.npz --output animazione.gif

Dipendenze:
    py -m pip install numpy matplotlib pillow
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    parser = argparse.ArgumentParser(
        description="Crea una GIF animata delle mappe Range-Doppler."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="dreamhat_dataset.npz",
        help="Percorso del dataset .npz (default: dreamhat_dataset.npz)",
    )
    parser.add_argument(
        "--output",
        default="range_doppler_animation.gif",
        help="Nome/percorso della GIF da creare.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frame al secondo della GIF (default: 10).",
    )
    parser.add_argument(
        "--rx",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Mostra un singolo ricevitore (0, 1 o 2). Default: mappa combinata.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset non trovato: {dataset_path.resolve()}\n"
            "Metti crea_gif.py nella stessa cartella di dreamhat_dataset.npz "
            "oppure passa il percorso completo del dataset."
        )

    with np.load(dataset_path) as data:
        if args.rx is None:
            if "range_doppler_power" not in data:
                raise KeyError("Nel dataset manca 'range_doppler_power'.")
            rd = data["range_doppler_power"]
            label = "Ricevitori combinati"
        else:
            if "range_doppler_power_per_rx" not in data:
                raise KeyError("Nel dataset manca 'range_doppler_power_per_rx'.")
            rd_all = data["range_doppler_power_per_rx"]
            rd = rd_all[:, args.rx, :, :]
            label = f"Ricevitore RX{args.rx + 1}"

        range_axis = data["range_m"] if "range_m" in data else None
        velocity_axis = data["velocity_m_s"] if "velocity_m_s" in data else None

    if rd.ndim != 3:
        raise ValueError(f"Mi aspettavo un array (frame, range, doppler), trovato {rd.shape}.")

    n_frames, n_range, n_doppler = rd.shape
    if n_frames == 0:
        raise ValueError("Il dataset non contiene frame.")

    # Converti la potenza in dB. Il piccolo termine evita log10(0).
    rd_db = 10.0 * np.log10(np.maximum(rd, 1e-12))

    # Scala cromatica fissa per tutti i frame, così i colori sono confrontabili.
    vmin = float(np.percentile(rd_db, 5))
    vmax = float(np.percentile(rd_db, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(
        rd_db[0],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )

    # Usa gli assi fisici se sono presenti nel file.
    if range_axis is not None and len(range_axis) == n_range:
        ax.set_ylabel("Distanza (m)")
        y_ticks = np.linspace(0, n_range - 1, min(6, n_range), dtype=int)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{range_axis[i]:.2f}" for i in y_ticks])
    else:
        ax.set_ylabel("Bin di distanza")

    if velocity_axis is not None and len(velocity_axis) == n_doppler:
        ax.set_xlabel("Velocità radiale (m/s)")
        x_ticks = np.linspace(0, n_doppler - 1, min(7, n_doppler), dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{velocity_axis[i]:.2f}" for i in x_ticks])
    else:
        ax.set_xlabel("Bin Doppler")

    title = ax.set_title(f"Range-Doppler — frame 1/{n_frames} — {label}")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Potenza (dB)")
    fig.tight_layout()

    def update(frame):
        image.set_data(rd_db[frame])
        title.set_text(f"Range-Doppler — frame {frame + 1}/{n_frames} — {label}")
        return image, title

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / max(1, args.fps),
        blit=False,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(output_path), writer=PillowWriter(fps=max(1, args.fps)))
    plt.close(fig)

    print(f"GIF creata: {output_path.resolve()}")
    print(f"Frame: {n_frames} | Dimensione di ogni mappa: {n_range} x {n_doppler}")
    print(f"Durata approssimativa: {n_frames / max(1, args.fps):.1f} secondi")


if __name__ == "__main__":
    main()
