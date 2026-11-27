import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from tkinter import Tk, filedialog
import numpy as np

# ============================================================
# CONFIGURAZIONE
# ============================================================

CSV_FILENAME = "dataset_manifest.csv"
DEFAULT_SUBJECT_ID = "sogg1"
RANDOM_SEED = 42

# Parole chiave nei nomi delle cartelle per dedurre la classe (label 1 vs 0)
FALL_KEYWORDS = ("cadut", "fall")
NON_FALL_KEYWORDS = (
    "non_cadut", "noncadut", "no_cadut", "nocadut",
    "non_fall", "nonfall", "no_fall", "nofall",
)

# Riconosce eventuali cartelle soggetto (es. sogg1, sogg2, sub01, utente_1)
SUBJECT_PATTERN = re.compile(
    r"^(sogg|soggetto|sub|subj|subject|utente|partecipante)[_-]?\d+$",
    re.IGNORECASE,
)


def select_root_folder() -> Path | None:
    """Apre la finestra di dialogo per selezionare la cartella radice del dataset."""
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Seleziona la cartella principale che contiene le sottocartelle delle attività"
    )
    root.destroy()
    return Path(folder) if folder else None


def classify_from_folders(rel_npz_path: Path) -> tuple[int, str, str]:
    """
    Estrae (label, subject_id, activity) dai nomi delle cartelle superiori,
    ignorando il nome casuale della cartella *_noise_mask_processed.
    """
    parent_parts = list(rel_npz_path.parent.parts)

    # Scarta l'ultima cartella se è quella generata dal processing (mmw_spi_..._noise_mask_processed)
    if parent_parts and (
        parent_parts[-1].endswith("_noise_mask_processed")
        or parent_parts[-1].endswith("_processed")
    ):
        semantic_folders = parent_parts[:-1]
    else:
        semantic_folders = parent_parts

    subject_id = DEFAULT_SUBJECT_ID
    activity_folders = []

    for folder in semantic_folders:
        clean_name = folder.strip()
        if SUBJECT_PATTERN.match(clean_name):
            subject_id = clean_name.lower()
        else:
            activity_folders.append(clean_name)

    activity = "/".join(activity_folders) if activity_folders else "sconosciuta"
    joined_lower = " ".join(activity_folders).lower()

    if any(kw in joined_lower for kw in NON_FALL_KEYWORDS):
        label = 0
    elif any(kw in joined_lower for kw in FALL_KEYWORDS):
        label = 1
    else:
        label = 0

    return label, subject_id, activity


def assign_splits(rows: list[dict], seed: int = 42) -> None:
    """
    Assegna 'train' (70%), 'val' (15%) e 'test' (15%).
    - Se ci sono >= 2 subject_id diversi, divide per soggetto (zero leakage tra persone).
    - Se c'è 1 solo soggetto, divide per singola registrazione stratificando per cartella (activity).
    """
    rng = np.random.default_rng(seed)
    unique_subjects = sorted({r["subject_id"] for r in rows})

    if len(unique_subjects) >= 2:
        shuffled_subs = rng.permutation(unique_subjects)
        n_sub = len(shuffled_subs)
        n_test = max(1, int(round(n_sub * 0.15)))
        n_val = max(1, int(round(n_sub * 0.15))) if n_sub >= 3 else 0

        test_subs = set(shuffled_subs[:n_test])
        val_subs = set(shuffled_subs[n_test : n_test + n_val])

        for r in rows:
            if r["subject_id"] in test_subs:
                r["split"] = "test"
            elif r["subject_id"] in val_subs:
                r["split"] = "val"
            else:
                r["split"] = "train"
        return

    # Caso 1 solo soggetto: split stratificato per cartella (activity) a livello di file .npz
    by_activity: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        by_activity[r["activity"]].append(idx)

    for _, indices in by_activity.items():
        perm = rng.permutation(indices)
        n = len(perm)
        if n == 1:
            rows[perm[0]]["split"] = "train"
        elif n == 2:
            rows[perm[0]]["split"] = "train"
            rows[perm[1]]["split"] = "val"
        else:
            n_test = max(1, int(round(n * 0.15)))
            n_val = max(1, int(round(n * 0.15)))
            for i in perm[:n_test]:
                rows[i]["split"] = "test"
            for i in perm[n_test : n_test + n_val]:
                rows[i]["split"] = "val"
            for i in perm[n_test + n_val :]:
                rows[i]["split"] = "train"


def main():
    root_dir = select_root_folder()
    if root_dir is None:
        print("Nessuna cartella selezionata. Operazione annullata.")
        return

    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname == "microdoppler_dataset.npz":
                full_path = Path(dirpath) / fname
                rel_path = full_path.relative_to(root_dir)
                label, subject_id, activity = classify_from_folders(rel_path)

                rows.append(
                    {
                        "npz_path": rel_path.as_posix(),
                        "label": label,
                        "subject_id": subject_id,
                        "activity": activity,
                        "split": "",
                        "start_s": "",
                        "end_s": "",
                    }
                )

    if not rows:
        print(f"Nessun file 'microdoppler_dataset.npz' trovato in {root_dir}")
        return

    rows.sort(key=lambda r: (r["label"], r["activity"], r["npz_path"]))
    assign_splits(rows, seed=RANDOM_SEED)

    output_csv = root_dir / CSV_FILENAME
    fieldnames = ["npz_path", "label", "subject_id", "activity", "split", "start_s", "end_s"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Riepilogo a schermo per verificare le cartelle lette
    print(f"\nCSV salvato in: {output_csv}")
    print(f"Totale registrazioni trovate: {len(rows)}\n")

    summary: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        summary[(r["label"], r["activity"])][r["split"]] += 1
        summary[(r["label"], r["activity"])]["tot"] += 1

    print(f"{'Label':<6} | {'Cartella (Activity)':<35} | {'Tot':>4} | {'Train':>5} | {'Val':>4} | {'Test':>4}")
    print("-" * 72)
    for (lbl, act), counts in sorted(summary.items()):
        print(
            f"{lbl:<6} | {act:<35} | {counts['tot']:>4} | "
            f"{counts['train']:>5} | {counts['val']:>4} | {counts['test']:>4}"
        )


if __name__ == "__main__":
    main()