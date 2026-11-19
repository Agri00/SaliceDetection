import sys
import os
import time  # <-- Importato per misurare il tempo trascorso
from datetime import datetime  # <-- Importato per avere gli orari in formato leggibile

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import utility.udp_streaming

IP_addr = "127.0.0.1"    # set the IP to receive data
port = 9575
cfg_dir = "../radar_config/config_3rx_5m"
filename = "data/mmw_spi"    # set None to disable saving data to file

# --- REGISTRAZIONE TEMPO DI AVVIO ---
start_time_monotonic = time.time()
start_time_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("-" * 50)
print(f"[{start_time_human}] -> Avvio della registrazione UDP / streaming...")
print(f"IP: {IP_addr} | Porta: {port} | Salvataggio su: {filename}")
print("-" * 50)

try:
    # Esegue lo streaming (questa funzione rimane attiva finché non la interrompi con Ctrl+C)
    utility.udp_streaming.main(IP_addr, port, cfg_dir, filename)

except KeyboardInterrupt:
    # Gestisce l'interruzione manuale da tastiera (es. quando premi Ctrl+C per fermare la registrazione)
    print("\n[!] Registrazione interrotta manualmente dall'utente (Ctrl+C).")

finally:
    # --- REGISTRAZIONE TEMPO DI CONCLUSIONE E CALCOLO DURATA ---
    end_time_monotonic = time.time()
    end_time_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    elapsed_time = end_time_monotonic - start_time_monotonic
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print("-" * 50)
    print(f"[{end_time_human}] -> Registrazione conclusa.")
    print(f"Orario di avvio:         {start_time_human}")
    print(f"Orario di conclusione:   {end_time_human}")
    print(f"Durata complessiva:      {elapsed_time:.2f} secondi ({minutes} min e {seconds:.2f} sec)")
    print("-" * 50)