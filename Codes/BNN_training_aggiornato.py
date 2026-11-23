import csv
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# PARAMETRI FINESTRA E BINARIZZAZIONE
# ============================================================

# 48 righe STFT = 16 frame radar = 1.6 secondi (a 10 Hz, 3 finestre STFT/frame)
WIN_STFT_STEPS = 48
STRIDE_STFT_STEPS = 12   # passo 0.4 s tra una finestra e la successiva
TARGET_SIZE = (24, 24)   # dimensione ingresso BNN


def format_binary_window_for_bnn(binary_2d: np.ndarray) -> np.ndarray:
    """
    Converte una finestra binaria {0, 1} di shape (WIN_STFT_STEPS, 64)
    in una matrice (24, 24) con valori in {-1.0, +1.0}.
    Usa adaptive_max_pool2d per non attenuare i picchi Doppler sottili.
    """
    t = torch.from_numpy(binary_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_max_pool2d(t, TARGET_SIZE).squeeze().numpy()
    return np.where(pooled > 0.5, 1.0, -1.0).astype(np.float32)


# ============================================================
# DATASET & DATA AUGMENTATION FISICA
# ============================================================

class MicroDopplerBNNDataset(Dataset):
    def __init__(
        self,
        power_windows: np.ndarray,
        binary_windows: np.ndarray,
        thresholds: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        """
        power_windows:  (N, 48, 64) float32 - potenza micro-Doppler lineare
        binary_windows: (N, 48, 64) uint8   - maschera binaria {0, 1} precalcolata
        thresholds:     (N, 64)     float32 - threshold_power della registrazione
        labels:         (N,)        int64   - 0 = Non-Fall, 1 = Fall
        """
        self.power_windows = power_windows
        self.binary_windows = binary_windows
        self.thresholds = thresholds
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def _augment_power_window(self, power: np.ndarray, thresh_1d: np.ndarray) -> np.ndarray:
        """
        Applica data augmentation nel dominio della potenza lineare PRIMA
        della soglia di rumore, mantenendo coerenza fisica con noise_reference.
        """
        x = power.copy()
        n_steps = x.shape[0]

        # 1. Time Warping (variazione velocità del movimento ±15%)
        speed_factor = random.uniform(0.85, 1.15)
        t_orig = np.arange(n_steps, dtype=np.float32)
        center = (n_steps - 1) / 2.0
        t_query = np.clip(center + (t_orig - center) * speed_factor, 0.0, n_steps - 1.0)
        idx0 = np.floor(t_query).astype(np.int32)
        idx1 = np.minimum(idx0 + 1, n_steps - 1)
        w = (t_query - idx0)[:, None]
        x = x[idx0] * (1.0 - w) + x[idx1] * w

        # 2. Temporal Shifting (±6 righe STFT = ±0.2 s, padding sui bordi)
        shift = random.randint(-6, 6)
        if shift > 0:
            x = np.pad(x[:-shift], ((shift, 0), (0, 0)), mode="edge")
        elif shift < 0:
            x = np.pad(x[-shift:], ((0, -shift), (0, 0)), mode="edge")

        # 3. Scaling RCS / distanza (simula corporature e distanze diverse dal radar)
        rcs_gain = random.uniform(0.75, 1.30)
        x = x * rcs_gain

        # 4. Fluttuazione rumore di fondo (rispetto alla soglia di rumore del bin)
        noise_scale = random.uniform(0.0, 0.12)
        if noise_scale > 0.0:
            noise = np.abs(np.random.normal(0.0, noise_scale, size=x.shape)).astype(np.float32)
            x = x + noise * thresh_1d[None, :]

        # Riapplica la maschera di rumore identica a process_with_noise_mask.py
        binary_aug = (x > thresh_1d[None, :]).astype(np.uint8)
        return binary_aug

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.augment:
            bin_win = self._augment_power_window(
                self.power_windows[idx],
                self.thresholds[idx],
            )
        else:
            bin_win = self.binary_windows[idx]

        mat_24x24 = format_binary_window_for_bnn(bin_win)
        tensor_x = torch.from_numpy(mat_24x24).unsqueeze(0)  # (1, 24, 24)
        tensor_y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor_x, tensor_y


# ============================================================
# ARCHITETTURA BNN (PESI E ATTIVAZIONI IN {-1, +1})
# ============================================================

class BinarySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1.0).float()


def binarize(x: torch.Tensor) -> torch.Tensor:
    return BinarySign.apply(x)


class BinaryConv2d(nn.Conv2d):
    def forward(self, x):
        return F.conv2d(
            x, binarize(self.weight), self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class BinaryLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, binarize(self.weight), self.bias)


class RadarFallBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = BinaryConv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = BinaryConv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)  # 24 -> 12 -> 6 -> 3
        self.fc1 = BinaryLinear(64 * 3 * 3, 64, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = binarize(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = binarize(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = binarize(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = binarize(self.bn4(self.fc1(x)))
        return self.fc2(x).squeeze(-1)


# ============================================================
# CARICAMENTO DA CSV E TRAINING
# ============================================================

def load_from_csv(
    csv_path: str | Path,
    base_dir: str | Path | None = None,
    target_split: str | None = None,
):
    """
    Carica le registrazioni dal file CSV.
    Se target_split è specificato ('train', 'val', 'test'), filtra solo quelle righe.
    """
    csv_path = Path(csv_path)
    root = Path(base_dir) if base_dir is not None else csv_path.parent

    all_power, all_binary, all_thresh, all_y, all_groups = [], [], [], [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        reader = csv.DictReader(f, dialect=dialect)

        for row in reader:
            row_split = (row.get("split") or "").strip().lower()
            if target_split and row_split != target_split.lower():
                continue

            label_str = (row.get("label") or "").strip()
            if label_str == "":
                raise ValueError(f"Etichetta mancante nel CSV per: {row.get('npz_path')}")

            label = int(label_str)
            group_id = (row.get("subject_id") or "default").strip()

            npz_file = Path(row["npz_path"].strip())
            if not npz_file.is_absolute():
                npz_file = root / npz_file

            data = np.load(npz_file)
            power = data["microdoppler_power"].astype(np.float32)
            binary = data["microdoppler_binary"].astype(np.uint8)
            thresh = data["threshold_power"].astype(np.float32)

            start_s = (row.get("start_s") or "").strip()
            end_s = (row.get("end_s") or "").strip()
            idx_start = int(float(start_s) * 30.0) if start_s else 0
            idx_end = int(float(end_s) * 30.0) if end_s else power.shape[0]

            power = power[idx_start:idx_end]
            binary = binary[idx_start:idx_end]

            if power.shape[0] < WIN_STFT_STEPS:
                print(f"[SKIP] {npz_file.name}: durata utile < {WIN_STFT_STEPS} righe STFT.")
                continue

            for start in range(0, power.shape[0] - WIN_STFT_STEPS + 1, STRIDE_STFT_STEPS):
                end = start + WIN_STFT_STEPS
                all_power.append(power[start:end])
                all_binary.append(binary[start:end])
                all_thresh.append(thresh)
                all_y.append(label)
                all_groups.append(group_id)

    return (
        np.asarray(all_power, dtype=np.float32),
        np.asarray(all_binary, dtype=np.uint8),
        np.asarray(all_thresh, dtype=np.float32),
        np.asarray(all_y, dtype=np.int64),
        np.asarray(all_groups),
    )


def train_model(
    csv_path: str | Path,
    epochs: int = 120,
    batch_size: int = 32,
    output_weights: str = "bnn_fall_detection.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Controlla se nel CSV è stata compilata manualmente la colonna 'split'
    tr_p, tr_b, tr_t, tr_y, _ = load_from_csv(csv_path, target_split="train")
    va_p, va_b, va_t, va_y, _ = load_from_csv(csv_path, target_split="val")

    if len(tr_y) > 0 and len(va_y) > 0:
        train_ds = MicroDopplerBNNDataset(tr_p, tr_b, tr_t, tr_y, augment=True)
        val_ds = MicroDopplerBNNDataset(va_p, va_b, va_t, va_y, augment=False)
        y_train = tr_y
    else:
        # Altrimenti carica tutto e divide automaticamente per subject_id
        power_w, binary_w, thresh_w, y, groups = load_from_csv(csv_path)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(power_w, y, groups))

        train_ds = MicroDopplerBNNDataset(
            power_w[train_idx], binary_w[train_idx], thresh_w[train_idx], y[train_idx],
            augment=True,
        )
        val_ds = MicroDopplerBNNDataset(
            power_w[val_idx], binary_w[val_idx], thresh_w[val_idx], y[val_idx],
            augment=False,
        )
        y_train = y[train_idx]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = RadarFallBNN().to(device)
    num_pos = max(1, int(y_train.sum()))
    num_neg = max(1, len(y_train) - num_pos)
    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p in model.parameters():
                    if p.ndim >= 2:
                        p.clamp_(-1.0, 1.0)
            running_loss += loss.item() * len(by)
        scheduler.step()

        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = (torch.sigmoid(model(bx)) >= 0.5).float()
                tp += ((preds == 1) & (by == 1)).sum().item()
                fp += ((preds == 1) & (by == 0)).sum().item()
                fn += ((preds == 0) & (by == 1)).sum().item()

        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), output_weights)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss: {running_loss/len(train_ds):.4f} | "
                f"Val Recall: {recall:.3f} | Val Precision: {precision:.3f} | F1: {f1:.3f}"
            )

    print(f"Modello salvato in {output_weights} (Miglior F1: {best_f1:.3f})")
    return model


if __name__ == "__main__":
    train_model("dataset_manifest.csv")
