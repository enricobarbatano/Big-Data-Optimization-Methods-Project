from pathlib import Path
import argparse
import sys

# ============================================================
# PATH FIX
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = CURRENT_DIR.parent                      

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch

from model import load_bundle

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = PROJECT_ROOT / "dataset_pytorch.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIM = 2000
DELIMITER = ","
SKIP_HEADER = 0   

DEFAULT_PROTOCOL = "a"   # a=cv10, b=701515, c=801010

# ============================================================
# UTILS
# ============================================================

def load_csv_dataset(csv_path: Path, delimiter: str = ",", skip_header: int = 0):
    csv_path = Path(csv_path).resolve()

    if not csv_path.exists():
        print("Dataset non trovato.")
        print(f"Path cercato: {csv_path}")
        raise FileNotFoundError(f"Dataset non trovato: {csv_path}")

    print(f"Caricamento dataset da: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=delimiter, skiprows=skip_header, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    X = data[:, :-1]
    y = data[:, -1:]

    if X.shape[1] != INPUT_DIM:
        raise ValueError(f"Attese {INPUT_DIM} feature, trovate {X.shape[1]}")

    print(f"Dataset caricato correttamente. Shape X = {X.shape}, shape y = {y.shape}")
    return X, y


def mse_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Validazione del modello surrogato f4")
    parser.add_argument(
        "--protocol",
        type=str,
        default=DEFAULT_PROTOCOL,
        choices=["a", "b", "c", "cv10", "701515", "801010"],
        help="a/cv10 | b/701515 | c/801010"
    )
    args = parser.parse_args()

    protocol = args.protocol.lower()
    if protocol == "a":
        protocol = "cv10"
    elif protocol == "b":
        protocol = "701515"
    elif protocol == "c":
        protocol = "801010"

    BUNDLE_DIR = CURRENT_DIR / "f4_surrogate_out" / protocol

    print(f"Working directory corrente: {Path.cwd()}")
    print(f"CSV_PATH configurato: {CSV_PATH.resolve()}")
    print(f"BUNDLE_DIR configurato: {BUNDLE_DIR.resolve()}")

    bundle, config = load_bundle(BUNDLE_DIR, device=DEVICE)
    X, y = load_csv_dataset(CSV_PATH, delimiter=DELIMITER, skip_header=SKIP_HEADER)

    x_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        y_pred = bundle.predict_f4(x_tensor).cpu().numpy()

    mse = mse_numpy(y, y_pred)
    mae = mae_numpy(y, y_pred)
    r2 = r2_numpy(y, y_pred)

    print("==============================")
    print("VALIDAZIONE MODELLO")
    print("==============================")
    print("Config:", config)
    print(f"MSE  = {mse:.6e}")
    print(f"MAE  = {mae:.6e}")
    print(f"R^2  = {r2:.6f}")

    outdir = BUNDLE_DIR / "predict_out"
    outdir.mkdir(parents=True, exist_ok=True)

    np.savez(outdir / "predictions_full_dataset.npz", y_true=y, y_pred=y_pred)
    print(f"[INFO] Predizioni salvate in: {outdir / 'predictions_full_dataset.npz'}")


if __name__ == "__main__":
    main()