import json
import sys
import os
import time  # <-- Importiamo il modulo time per calcolare le durate
from datetime import datetime  # <-- Importiamo datetime per mostrare l'orario leggibile
import numpy as np
import matplotlib.pyplot as plt

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from utility.helper import find_setting_in_directory
from utility.mmw_cube_proc_v0 import CubeProcessor


def main(data_fn, proc_save_dir, plots, cfg_dir, num_angle_bins):
    setting_fn = find_setting_in_directory(cfg_dir)
    with open(setting_fn, 'r') as file:
        setting = json.load(file)
    mmw_proc = CubeProcessor(setting, num_azimuth_bin=num_angle_bins, num_elevation_bin=num_angle_bins)

    if proc_save_dir:
        os.makedirs(proc_save_dir, exist_ok=True)

    # --- REGISTRAZIONE TEMPO DI AVVIO ---
    start_time_monotonic = time.time()  # Ottimo per calcolare la durata esatta in secondi
    start_time_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Formato leggibile
    
    print(f"[{start_time_human}] -> Avvio del processo di elaborazione frame...")

    frame_count = 0  # Contatore opzionale per sapere quanti frame sono stati processati

    # Open the file in binary read mode
    with open(data_fn, "rb") as file:
        while True:
            version_bytes = file.read(4)
            if not version_bytes:
                break
            version = int.from_bytes(version_bytes, byteorder='little', signed=False)  # Read the first N bytes
            if version == 0:
                seq = int.from_bytes(file.read(4), byteorder='little', signed=False)
                data_len = int.from_bytes(file.read(4), byteorder='little', signed=False)
                mmw_data = file.read(data_len)
                mmw_proc.process_raw_data(mmw_data)

                frame_count += 1
                
                print(f"seq: {seq}")
                for plot in plots:
                    axis_0_name, axis_1_name = plot
                    img = mmw_proc.vis_2d(axis_0_name, axis_1_name)
                    img = np.log10(img)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.savefig(os.path.join(proc_save_dir, str(seq) + "_" + axis_0_name + "_" + axis_1_name +".jpg"), bbox_inches='tight', pad_inches=0)
                    plt.close()


# --- REGISTRAZIONE TEMPO DI CONCLUSIONE ---
    end_time_monotonic = time.time()
    end_time_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calcolo della durata complessiva in secondi
    elapsed_time = end_time_monotonic - start_time_monotonic
    
    # Formattazione della durata in minuti e secondi (utile se il processo dura molto)
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print("-" * 50)
    print(f"[{end_time_human}] -> Elaborazione conclusa con successo!")
    print(f"Totale frame processati: {frame_count}")
    print(f"Orario di avvio:         {start_time_human}")
    print(f"Orario di conclusione:   {end_time_human}")
    print(f"Durata complessiva:      {elapsed_time:.2f} secondi ({minutes} min e {seconds:.2f} sec)")
    print("-" * 50)

if __name__ == '__main__':
    data_fn = "data/mmw_spi_20241119_081114_994.bin"
        
    if len(sys.argv) > 1:
        data_fn = sys.argv[1]
    else:
        data_fn = "data/mmw_spi_20241119_081114_994.bin"
        print("No file provided in terminal. Using default fallback.")
            
    file_basename = os.path.splitext(os.path.basename(data_fn))[0]
    
    proc_save_dir = os.path.join("mmw_proc/", file_basename) + "/"
    
    
    plots = [("Range", "Doppler")]
    cfg_dir = "../radar_config/config_3rx_5m"
    num_angle_bins = 16
    main(data_fn, proc_save_dir, plots, cfg_dir, num_angle_bins)
