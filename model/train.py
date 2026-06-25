# per addestrare il modello scelto, rinominalo in model.py

import sys
from pathlib import Path
import argparse
import json

# ============================================================
# PATH FIX
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch

from algorithms.adam import CustomAdam
from model import F4SurrogateNet, save_bundle

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = PROJECT_ROOT / "dataset_unito.csv"  
OUTDIR_BASE = CURRENT_DIR / "f4_surrogate_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIM = 2000
OUTPUT_DIM = 1
HIDDEN_DIMS = [1024, 516, 256, 128, 64, 32, 16]
ACTIVATION = "gelu"

SEED = 42
EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-6
PATIENCE = 30

DELIMITER = ","
SKIP_HEADER = 0

# Protocolli:
# a = cv10
# b = 70/15/15
# c = 80/10/10
DEFAULT_PROTOCOL = "a"

# ============================================================
# UTILS DATASET
# ============================================================

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        print(f"Shape dataset caricata: {data.shape}")
        raise ValueError(f"Attese {INPUT_DIM} feature, trovate {X.shape[1]}")

    print(f"Dataset caricato correttamente. Shape X = {X.shape}, shape y = {y.shape}")
    return X, y


def split_dataset(X: np.ndarray, y: np.ndarray,
                  train_ratio: float, val_ratio: float, test_ratio: float,
                  seed: int = 42):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    X = X[idx]
    y = y[idx]

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def standardize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-12] = 1.0

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-12] = 1.0

    X_train_s = (X_train - x_mean) / x_std
    X_val_s   = (X_val - x_mean) / x_std
    X_test_s  = (X_test - x_mean) / x_std

    y_train_s = (y_train - y_mean) / y_std
    y_val_s   = (y_val - y_mean) / y_std
    y_test_s  = (y_test - y_mean) / y_std

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, stats


def standardize_full(X, y):
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True)
    x_std[x_std < 1e-12] = 1.0

    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std[y_std < 1e-12] = 1.0

    Xs = (X - x_mean) / x_std
    ys = (y - y_mean) / y_std

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return Xs, ys, stats


def batch_iterator(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = idx[start:end]
        yield X[ids], y[ids]

# ============================================================
# LOSS + METRICHE
# ============================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def l2_regularization(model: torch.nn.Module) -> torch.Tensor:
    reg = torch.tensor(0.0, dtype=torch.float32, device=next(model.parameters()).device)
    for p in model.parameters():
        reg = reg + torch.sum(p * p)
    return reg


def mse_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

# ============================================================
# TRAIN
# ============================================================

def train_model(model: torch.nn.Module,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray):
    model.to(DEVICE)
    optimizer = CustomAdam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

    best_val = np.inf
    best_state = None
    patience_counter = 0
    train_history = []
    val_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb_np, yb_np in batch_iterator(X_train, y_train, BATCH_SIZE, shuffle=True):
            xb = torch.tensor(xb_np, dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(yb_np, dtype=torch.float32, device=DEVICE)

            pred = model(xb)
            loss = mse_loss(pred, yb) + WEIGHT_DECAY * l2_regularization(model)

            loss.backward()
            optimizer.step()

            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # UNA sola volta per batch
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
            yv = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
            pv = model(xv)
            val_loss = mse_loss(pv, yv).item()

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch+1:4d} | train_loss = {train_loss:.6e} | val_loss = {val_loss:.6e}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping all'epoca {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_history, val_history, best_val


def predict_numpy(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        pred = model(xt).cpu().numpy()
    return pred

# ============================================================
# PROTOCOLLI
# ============================================================

def train_and_evaluate_split(X, y, train_ratio, val_ratio, test_ratio):
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X, y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=SEED,
    )

    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, stats = standardize_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    model = F4SurrogateNet(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=OUTPUT_DIM,
        activation=ACTIVATION,
    )

    train_hist, val_hist, best_val = train_model(model, X_train_s, y_train_s, X_val_s, y_val_s)

    y_pred_test_s = predict_numpy(model, X_test_s)
    y_pred_test = y_pred_test_s * stats["y_std"] + stats["y_mean"]

    metrics = {
        "mse": mse_numpy(y_test, y_pred_test),
        "mae": mae_numpy(y_test, y_pred_test),
        "r2": r2_numpy(y_test, y_pred_test),
        "best_val": float(best_val),
    }

    return model, train_hist, val_hist, stats, y_test, y_pred_test, metrics



def save_training_artifacts(outdir, protocol_name, model, train_hist, val_hist, stats,
                            y_test=None, y_pred_test=None, extra_metrics=None):
    outdir.mkdir(parents=True, exist_ok=True)

    save_bundle(
        model=model,
        outdir=outdir,
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=OUTPUT_DIM,
        activation=ACTIVATION,
    )

    np.savez(outdir / "f4_surrogate_loss_curves.npz",
             train_loss=np.array(train_hist),
             val_loss=np.array(val_hist))

    if y_test is not None and y_pred_test is not None:
        np.savez(outdir / "f4_surrogate_test_predictions.npz",
                 y_true=y_test,
                 y_pred=y_pred_test)

    payload = {
        "protocol": protocol_name,
        "input_dim": INPUT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "activation": ACTIVATION,
    }
    if extra_metrics is not None:
        for k, v in extra_metrics.items():
            try:
                payload[k] = float(v)
            except Exception:
                payload[k] = v

    with open(outdir / "training_protocol.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Modello e statistiche salvati in: {outdir}")


def run_protocol_split(X, y, train_ratio, val_ratio, test_ratio, protocol_name, outdir):
    model, train_hist, val_hist, stats, y_test, y_pred_test, metrics = train_and_evaluate_split(
        X, y, train_ratio, val_ratio, test_ratio
    )

    print("\n==============================")
    print(f"RISULTATI TEST - protocol={protocol_name}")
    print("==============================")
    print(f"MSE  = {metrics['mse']:.6e}")
    print(f"MAE  = {metrics['mae']:.6e}")
    print(f"R^2  = {metrics['r2']:.6f}")

    save_training_artifacts(outdir, protocol_name, model, train_hist, val_hist, stats,
                            y_test, y_pred_test, metrics)

def run_protocol_cv10(X, y, outdir):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)

    folds = np.array_split(idx, 10)
    cv_rows = []
    best_fold_val = np.inf

    for fold_id in range(10):
        test_idx = folds[fold_id]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fold_id])

        n_train = len(train_idx)
        n_val = max(1, int(0.15 * n_train))
        val_idx = train_idx[:n_val]
        train2_idx = train_idx[n_val:]

        X_train, y_train = X[train2_idx], y[train2_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, _ = standardize_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        model = F4SurrogateNet(
            input_dim=INPUT_DIM,
            hidden_dims=HIDDEN_DIMS,
            output_dim=OUTPUT_DIM,
            activation=ACTIVATION,
        )

        _, _, best_val = train_model(model, X_train_s, y_train_s, X_val_s, y_val_s)
        y_pred_test_s = predict_numpy(model, X_test_s)

        y_mean = y_train.mean(axis=0, keepdims=True)
        y_std = y_train.std(axis=0, keepdims=True)
        y_std[y_std < 1e-12] = 1.0
        y_pred_test = y_pred_test_s * y_std + y_mean

        row = {
            "fold": fold_id + 1,
            "val_loss_best": float(best_val),
            "test_mse": mse_numpy(y_test, y_pred_test),
            "test_mae": mae_numpy(y_test, y_pred_test),
            "test_r2": r2_numpy(y_test, y_pred_test),
        }
        cv_rows.append(row)
        print(f"[FOLD {fold_id+1}] {row}")

        if best_val < best_fold_val:
            best_fold_val = best_val

    # retrain finale su tutto il dataset
    Xs, ys, stats = standardize_full(X, y)
    model_final = F4SurrogateNet(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=OUTPUT_DIM,
        activation=ACTIVATION,
    )

    n_val_final = max(1, int(0.10 * Xs.shape[0]))
    X_val_f, y_val_f = Xs[:n_val_final], ys[:n_val_final]
    X_train_f, y_train_f = Xs[n_val_final:], ys[n_val_final:]

    train_hist, val_hist, _ = train_model(model_final, X_train_f, y_train_f, X_val_f, y_val_f)

    save_training_artifacts(
        outdir,
        "cv10",
        model_final,
        train_hist,
        val_hist,
        stats,
        extra_metrics={"best_fold_val": float(best_fold_val)}
    )

    np.savez(outdir / "cv10_results.npz", rows=np.array(cv_rows, dtype=object))
    print(f"[INFO] CV10 completata; modello finale salvato in: {outdir}")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Training surrogato f4 con protocollo parametrico")
    parser.add_argument(
        "--protocol",
        type=str,
        default=DEFAULT_PROTOCOL,
        choices=["a", "b", "c", "cv10", "701515", "801010"],
        help="a/cv10 | b/701515 | c/801010"
    )
    args = parser.parse_args()

    set_seed(SEED)
    print(f"[DEBUG] Working directory corrente: {Path.cwd()}")
    print(f"[DEBUG] CSV_PATH configurato: {CSV_PATH.resolve()}")

    X, y = load_csv_dataset(CSV_PATH, delimiter=DELIMITER, skip_header=SKIP_HEADER)

    protocol = args.protocol.lower()
    if protocol == "a":
        protocol = "cv10"
    elif protocol == "b":
        protocol = "701515"
    elif protocol == "c":
        protocol = "801010"

    outdir = OUTDIR_BASE / protocol

    if protocol == "cv10":
        run_protocol_cv10(X, y, outdir)
    elif protocol == "701515":
        run_protocol_split(X, y, 0.70, 0.15, 0.15, protocol, outdir)
    elif protocol == "801010":
        run_protocol_split(X, y, 0.80, 0.10, 0.10, protocol, outdir)
    else:
        raise ValueError(f"Protocollo non supportato: {protocol}")



if __name__ == "__main__":
    main()
