
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from pathlib import Path

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("--output", default="microdoppler.gif")
    parser.add_argument("--rx", type=int, default=-1,
                        help="-1 = media RX, 0/1/2 = singolo ricevitore")
    parser.add_argument("--fps", type=int, default=10)

    args = parser.parse_args()

    data = np.load(args.dataset)

    power = data["micro_doppler_power_per_rx"]

    # Shape originale:
    # (frame, finestra STFT, Doppler, range, RX)

    if args.rx == -1:
        power = power.mean(axis=-1)
    else:
        power = power[..., args.rx]

    velocity = data["velocity_m_s"]

    n_frames, n_windows, n_doppler, n_range = power.shape

    # Conversione in dB
    power_db = 10 * np.log10(power + 1e-12)

    # Limiti cromatici globali
    vmin = np.percentile(power_db, 5)
    vmax = np.percentile(power_db, 99.5)

    fig, ax = plt.subplots(figsize=(8, 6))

    first = power_db[0, 0]

    image = ax.imshow(
        first,
        aspect="auto",
        origin="lower",
        extent=[
            0, n_range,
            velocity[0], velocity[-1]
        ],
        cmap="inferno",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel("Range bin")
    ax.set_ylabel("Velocità radiale (m/s)")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Potenza (dB)")

    title = ax.set_title("")

    def update(index):

        frame = index // n_windows
        window = index % n_windows

        image.set_data(power_db[frame, window])

        title.set_text(
            f"Frame {frame + 1}/{n_frames} | "
            f"STFT window {window + 1}/{n_windows}"
        )

        return image, title

    total_images = n_frames * n_windows

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_images,
        interval=1000 / args.fps,
        blit=False
    )

    ani.save(
        args.output,
        writer=animation.PillowWriter(fps=args.fps)
    )

    plt.close(fig)

    print("GIF salvata:", Path(args.output).resolve())
    print("Numero immagini:", total_images)


if __name__ == "__main__":
    main()
