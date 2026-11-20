
import os
import numpy as np
from tkinter import Tk, filedialog

from radar_common import (
    decode_bin_file,
    calculate_microdoppler,
    convert_to_db,
    normalize_minmax,
    calculate_velocity_axis,
    save_plot
)


# ============================================================
# PARAMETRI BINARIZZAZIONE
# ============================================================

# Margine sopra il profilo di rumore di riferimento.

NOISE_MARGIN_DB = 6

# ============================================================
# OUTPUT
# ============================================================

SAVE_PLOTS = True
SAVE_NPZ = True


def main():

    root = Tk()
    root.withdraw()

    bin_file = filedialog.askopenfilename(
        title="Seleziona la registrazione CON movimento",
        filetypes=[
            ("BIN files", "*.bin"),
            ("All files", "*.*")
        ]
    )

    if not bin_file:
        root.destroy()
        print("Nessun file BIN selezionato.")
        return

    reference_file = filedialog.askopenfilename(
        title="Seleziona noise_reference.npz",
        filetypes=[
            ("NPZ files", "*.npz"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not reference_file:
        print("Nessun riferimento rumore selezionato.")
        return

    # ========================================================
    # CARICAMENTO RIFERIMENTO
    # ========================================================

    print("\nCaricamento riferimento rumore...")

    reference_data = np.load(reference_file)

    noise_reference = reference_data[
        "noise_reference"
    ]

    if noise_reference.shape != (64,):

        raise ValueError(
            "Il riferimento deve contenere 64 bin Doppler."
        )

    print("Riferimento caricato.")
    print("Numero registrazioni di riferimento:",
          int(reference_data["number_of_recordings"]))

    # ========================================================
    # DECODIFICA E PROCESSING
    # ========================================================

    adc_data, sequences = decode_bin_file(bin_file)

    (
        microdoppler_power,
        frame_indices,
        time_values
    ) = calculate_microdoppler(
        adc_data,
        sequences
    )

    microdoppler_db = convert_to_db(
        microdoppler_power
    )

    normalized = normalize_minmax(
        microdoppler_db
    )

    # ========================================================
    # MASCHERA DI RUMORE
    # ========================================================

    print("\nApplicazione maschera di rumore...")

    margin_linear = 10.0 ** (
        NOISE_MARGIN_DB / 10.0
    )

    # Soglia per ogni bin Doppler.
    # Viene replicata automaticamente su tutte le righe temporali.

    threshold_power = (
        noise_reference * margin_linear
    )

    threshold_matrix = np.broadcast_to(
        threshold_power,
        microdoppler_power.shape
    )

    binary = (
        microdoppler_power > threshold_matrix
    ).astype(np.uint8)

    # Conversione in dB per visualizzazione

    noise_reference_db = 10.0 * np.log10(
        noise_reference + 1e-12
    )

    threshold_db = 10.0 * np.log10(
        threshold_power + 1e-12
    )

    # ========================================================
    # ASSE VELOCITA'
    # ========================================================

    velocity = calculate_velocity_axis()

    # ========================================================
    # CARTELLA OUTPUT
    # ========================================================

    output_dir = os.path.join(
        os.path.dirname(bin_file),
        os.path.splitext(
            os.path.basename(bin_file)
        )[0] + "_noise_mask_processed"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

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
            "viridis"
        )

        save_plot(
            normalized,
            time_values,
            velocity,
            "Micro-Doppler normalizzato Min-Max",
            "microdoppler_normalized.png",
            output_dir,
            "Valore normalizzato",
            "viridis"
        )

        save_plot(
            binary,
            time_values,
            velocity,
            "Micro-Doppler binarizzato - Noise Reference",
            "microdoppler_binary.png",
            output_dir,
            "Stato",
            "gray"
        )

        save_plot(
            np.tile(
                noise_reference_db,
                (len(time_values), 1)
            ),
            time_values,
            velocity,
            "Profilo di rumore di riferimento",
            "noise_reference.png",
            output_dir,
            "Potenza rumore (dB)",
            "viridis"
        )

        save_plot(
            np.tile(
                threshold_db,
                (len(time_values), 1)
            ),
            time_values,
            velocity,
            "Soglia di binarizzazione",
            "noise_threshold.png",
            output_dir,
            "Soglia (dB)",
            "viridis"
        )

    # ========================================================
    # SALVATAGGIO NPZ
    # ========================================================

    if SAVE_NPZ:

        output_path = os.path.join(
            output_dir,
            "microdoppler_dataset.npz"
        )

        np.savez_compressed(
            output_path,

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

            noise_margin_db=NOISE_MARGIN_DB
        )

        print("\nDataset salvato:", output_path)

    # ========================================================
    # RIEPILOGO
    # ========================================================

    print("\n===================================")
    print("PROCESSING COMPLETATO")
    print("===================================")

    print("Frame radar:", len(adc_data))
    print("Frame micro-Doppler:", len(binary))
    print("Bin Doppler:", binary.shape[1])

    print(
        "Percentuale celle attive:",
        round(100 * np.mean(binary), 2),
        "%"
    )

    print("Margine rumore:", NOISE_MARGIN_DB, "dB")
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
