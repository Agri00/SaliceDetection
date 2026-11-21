import os
import sys
import traceback
from tkinter import Tk, filedialog, messagebox
import numpy as np

# Forza Python a cercare i moduli nella stessa cartella di questo script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from radar_common import (
    decode_bin_file,
    calculate_microdoppler,
    convert_to_db,
    normalize_minmax,
    calculate_velocity_axis,
    save_plot,
)


# ============================================================
# PARAMETRI BINARIZZAZIONE
# ============================================================

# Margine sopra il profilo di rumore di riferimento.
NOISE_MARGIN_DB = 3

# ============================================================
# OUTPUT
# ============================================================

SAVE_PLOTS = True
SAVE_NPZ = True


def select_bin_files(root: Tk) -> list[str]:
    """Consente di selezionare più file .bin oppure un'intera cartella."""
    use_folder = messagebox.askyesno(
        title="Modalità selezione registrazioni",
        message=(
            "Vuoi selezionare un'intera cartella (incluse eventuali sottocartelle)?\n\n"
            "• Sì = seleziona una cartella e processa tutti i .bin contenuti\n"
            "• No = seleziona manualmente uno o più file .bin"
        ),
    )

    if use_folder:
        folder = filedialog.askdirectory(
            title="Seleziona la cartella con le registrazioni .bin"
        )
        if not folder:
            return []

        bin_files = []
        for dirpath, _, filenames in os.walk(folder):
            for fname in filenames:
                if fname.lower().endswith(".bin"):
                    bin_files.append(os.path.join(dirpath, fname))
        return sorted(bin_files)

    files = filedialog.askopenfilenames(
        title="Seleziona le registrazioni CON movimento",
        filetypes=[
            ("BIN files", "*.bin"),
            ("All files", "*.*"),
        ],
    )
    return list(files)


def process_single_recording(
    bin_file: str,
    noise_reference: np.ndarray,
    threshold_power: np.ndarray,
    noise_reference_db: np.ndarray,
    threshold_db: np.ndarray,
    velocity: np.ndarray,
) -> dict:
    """Esegue decodifica, micro-Doppler, maschera di rumore e salvataggio su un singolo .bin."""
    adc_data, sequences = decode_bin_file(bin_file)

    (
        microdoppler_power,
        frame_indices,
        time_values,
    ) = calculate_microdoppler(
        adc_data,
        sequences,
    )

    microdoppler_db = convert_to_db(microdoppler_power)
    normalized = normalize_minmax(microdoppler_db)

    # ========================================================
    # MASCHERA DI RUMORE
    # ========================================================

    threshold_matrix = np.broadcast_to(
        threshold_power,
        microdoppler_power.shape,
    )

    binary = (
        microdoppler_power > threshold_matrix
    ).astype(np.uint8)

    # ========================================================
    # CARTELLA OUTPUT
    # ========================================================

    output_dir = os.path.join(
        os.path.dirname(bin_file),
        os.path.splitext(os.path.basename(bin_file))[0] + "_noise_mask_processed",
    )
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================
    # GRAFICI
    # ========================================================

    if SAVE_PLOTS:
        save_plot(
            microdoppler_db,
            time_values,
            velocity,
            "Micro-Doppler originale (dB)",
            "microdoppler_db.png",
            output_dir,
            "Potenza (dB)",
            "viridis",
        )

        save_plot(
            normalized,
            time_values,
            velocity,
            "Micro-Doppler normalizzato Min-Max",
            "microdoppler_normalized.png",
            output_dir,
            "Valore normalizzato",
            "viridis",
        )

        save_plot(
            binary,
            time_values,
            velocity,
            "Micro-Doppler binarizzato - Noise Reference",
            "microdoppler_binary.png",
            output_dir,
            "Stato",
            "gray",
        )

        save_plot(
            np.tile(noise_reference_db, (len(time_values), 1)),
            time_values,
            velocity,
            "Profilo di rumore di riferimento",
            "noise_reference.png",
            output_dir,
            "Potenza rumore (dB)",
            "viridis",
        )

        save_plot(
            np.tile(threshold_db, (len(time_values), 1)),
            time_values,
            velocity,
            "Soglia di binarizzazione",
            "noise_threshold.png",
            output_dir,
            "Soglia (dB)",
            "viridis",
        )

    # ========================================================
    # SALVATAGGIO NPZ
    # ========================================================

    npz_path = None
    if SAVE_NPZ:
        npz_path = os.path.join(
            output_dir,
            "microdoppler_dataset.npz",
        )

        np.savez_compressed(
            npz_path,
            microdoppler_power=microdoppler_power,
            microdoppler_db=microdoppler_db,
            microdoppler_normalized=normalized,
            microdoppler_binary=binary,
            noise_reference=noise_reference,
            noise_reference_db=noise_reference_db,
            threshold_power=threshold_power,
            threshold_db=threshold_db,
            frame_indices=frame_indices,
            time_values=time_values,
            velocity=velocity,
            noise_margin_db=NOISE_MARGIN_DB,
        )

    stats = {
        "file": os.path.basename(bin_file),
        "radar_frames": len(adc_data),
        "stft_frames": len(binary),
        "active_pct": round(100.0 * float(np.mean(binary)), 2),
        "output_dir": output_dir,
        "npz_path": npz_path,
    }

    # Rilascio esplicito della memoria tra un file e il successivo
    del adc_data, microdoppler_power, microdoppler_db, normalized, binary

    return stats


def main():
    root = Tk()
    root.withdraw()

    bin_files = select_bin_files(root)

    if not bin_files:
        root.destroy()
        print("Nessun file BIN selezionato.")
        return

    reference_file = filedialog.askopenfilename(
        title="Seleziona noise_reference.npz",
        filetypes=[
            ("NPZ files", "*.npz"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not reference_file:
        print("Nessun riferimento rumore selezionato.")
        return

    # ========================================================
    # CARICAMENTO RIFERIMENTO E PRECALCOLO SOGLIE
    # ========================================================

    print("\nCaricamento riferimento rumore...")
    reference_data = np.load(reference_file)
    noise_reference = reference_data["noise_reference"]

    if noise_reference.shape != (64,):
        raise ValueError("Il riferimento deve contenere 64 bin Doppler.")

    margin_linear = 10.0 ** (NOISE_MARGIN_DB / 10.0)
    threshold_power = noise_reference * margin_linear
    noise_reference_db = 10.0 * np.log10(noise_reference + 1e-12)
    threshold_db = 10.0 * np.log10(threshold_power + 1e-12)
    velocity = calculate_velocity_axis()

    print("Riferimento caricato.")
    print("Numero registrazioni di riferimento:", int(reference_data["number_of_recordings"]))
    print(f"Registrazioni da elaborare: {len(bin_files)}")

    # ========================================================
    # ELABORAZIONE BATCH DI TUTTI I FILE
    # ========================================================

    completed = []
    failed = []

    for idx, bin_file in enumerate(bin_files, start=1):
        print("\n===================================================")
        print(f"[{idx}/{len(bin_files)}] Elaborazione: {os.path.basename(bin_file)}")
        print("===================================================")

        try:
            stats = process_single_recording(
                bin_file=bin_file,
                noise_reference=noise_reference,
                threshold_power=threshold_power,
                noise_reference_db=noise_reference_db,
                threshold_db=threshold_db,
                velocity=velocity,
            )
            completed.append(stats)
            print(
                f"Completato | Frame radar: {stats['radar_frames']} | "
                f"Frame micro-Doppler: {stats['stft_frames']} | "
                f"Celle attive: {stats['active_pct']}%"
            )
            if stats["npz_path"]:
                print("Dataset salvato:", stats["npz_path"])

        except Exception as exc:
            failed.append((bin_file, str(exc)))
            print(f"ERRORE su {os.path.basename(bin_file)}: {exc}")
            traceback.print_exc()

    # ========================================================
    # RIEPILOGO FINALE BATCH
    # ========================================================

    print("\n===================================================")
    print("RIEPILOGO BATCH PROCESSING")
    print("===================================================")
    print(f"File totali trovati:  {len(bin_files)}")
    print(f"Elaborati con successo: {len(completed)}")
    print(f"Errori:               {len(failed)}")
    print(f"Margine rumore:       {NOISE_MARGIN_DB} dB")

    if completed:
        print("\nDettaglio registrazioni elaborate:")
        for item in completed:
            print(
                f"  - {item['file']}: {item['stft_frames']} frame STFT, "
                f"{item['active_pct']}% celle attive"
            )

    if failed:
        print("\nRegistrazioni con errore:")
        for path, err in failed:
            print(f"  - {path}: {err}")


if __name__ == "__main__":
    main()
