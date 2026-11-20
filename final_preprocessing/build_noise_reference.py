
import os
import numpy as np
from tkinter import Tk, filedialog

from radar_common import (
    decode_bin_file,
    calculate_microdoppler
)


# ============================================================
# PARAMETRI MASCHERA
# ============================================================

# Percentile calcolato nel tempo, separatamente
# per ogni bin Doppler e per ogni registrazione.

NOISE_PERCENTILE = 95

# Aggregazione dei profili delle diverse registrazioni:
# "median" oppure "percentile"

REFERENCE_AGGREGATION = "percentile"

# Utilizzato solo se REFERENCE_AGGREGATION = "percentile"

REFERENCE_PERCENTILE = 90


def main():

    root = Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(
        title="Seleziona le registrazioni SENZA movimento",
        filetypes=[
            ("BIN files", "*.bin"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not files:
        print("Nessun file selezionato.")
        return

    print("\nRegistrazioni selezionate:", len(files))

    recording_profiles = []

    for file_index, file_path in enumerate(files, start=1):

        print("\n===================================")
        print(f"Registrazione {file_index}/{len(files)}")
        print(os.path.basename(file_path))
        print("===================================")

        adc_data, sequences = decode_bin_file(file_path)

        microdoppler_power, _, _ = calculate_microdoppler(
            adc_data,
            sequences
        )

        # Un profilo per registrazione:
        # 64 valori, uno per ogni bin Doppler.

        profile = np.percentile(
            microdoppler_power,
            NOISE_PERCENTILE,
            axis=0
        )

        recording_profiles.append(profile)

        print("Profilo rumore calcolato.")

        del adc_data
        del microdoppler_power

    recording_profiles = np.stack(
        recording_profiles,
        axis=0
    )

    if REFERENCE_AGGREGATION.lower() == "median":

        noise_reference = np.median(
            recording_profiles,
            axis=0
        )

    elif REFERENCE_AGGREGATION.lower() == "percentile":

        noise_reference = np.percentile(
            recording_profiles,
            REFERENCE_PERCENTILE,
            axis=0
        )

    else:

        raise ValueError(
            "REFERENCE_AGGREGATION deve essere "
            "'median' oppure 'percentile'."
        )

    # Salvataggio accanto al primo file selezionato

    output_dir = os.path.dirname(files[0])

    output_path = os.path.join(
        output_dir,
        "noise_reference.npz"
    )

    np.savez_compressed(
        output_path,

        noise_reference=noise_reference,
        recording_profiles=recording_profiles,

        noise_percentile=NOISE_PERCENTILE,
        reference_aggregation=REFERENCE_AGGREGATION,
        reference_percentile=REFERENCE_PERCENTILE,

        number_of_recordings=len(files)
    )

    print("\n===================================")
    print("RIFERIMENTO RUMORE CREATO")
    print("===================================")

    print("Registrazioni utilizzate:", len(files))
    print("Profilo:", noise_reference.shape)
    print("File:", output_path)


if __name__ == "__main__":
    main()
