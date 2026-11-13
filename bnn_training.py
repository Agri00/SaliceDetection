import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


# Ottiene file nella cartella /decoded_data
class GifDataset(Dataset):
    def __init__(self, data_dir="./decoded_data", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Filtra e seleziona solo i file con estensione .gif nella cartella
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.gif')]
        
        if len(self.image_files) == 0:
            print(f"Attenzione: Nessun file .gif trovato nella cartella '{data_dir}'.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Percorso completo del file GIF
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Apre l'immagine GIF con PIL e la converte in RGB (necessario per i modelli standard)
        image = Image.open(img_path).convert('RGB')
        
        # Applicazione della pipeline di trasformazione/preprocessing se definita
        if self.transform:
            image = self.transform(image)
            
        # Nota: Qui puoi associare un'etichetta (label) se stai facendo classificazione, 
        # ad esempio estraendola dal nome del file (es. "classe_1_001.gif").
        # Per ora restituiamo solo l'immagine trasformata come tensore.
        return image