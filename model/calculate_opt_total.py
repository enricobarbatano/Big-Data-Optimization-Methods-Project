import os
import sys
from pathlib import Path

# ============================================================
# CONFIG PER RIPETIBILITA'
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch

# ============================================================
# PATH FIX
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent          
PROJECT_ROOT = CURRENT_DIR.parent                      

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ============================================================
# IMPORT
# ============================================================
from algorithms.GDM import gradient_descent_momentum
from model import load_bundle
from algorithms.bfgs import lbfgs_optimizer
from functions.obj import f1, f2, f3
from functions.gradients import grad_total

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cpu"

BUNDLE_DIR = CURRENT_DIR / "f4_surrogate_out" / "cv10"

X_REF_TXT_PATH = CURRENT_DIR / "x_ref.txt"

INPUT_DIM = 2000
M = 1000
BOUND = 300.0

# ampiezza massima della perturbazione locale attorno al punto fisso
DELTA_MAX = 2.5

# seed globale per ripetibilita'
SEED = 42

# Se True, impedisce al surrogato di predire f4 negative.
CLAMP_F4_TO_ZERO = True

# ============================================================
# LETTURA x_ref DA FILE TXT
# ============================================================

def load_reference_from_txt(txt_path: Path) -> np.ndarray:
    if not txt_path.exists():
        raise FileNotFoundError(f"File x_ref non trovato: {txt_path}")

    vals = [float(x) for x in txt_path.read_text(encoding="utf-8").split() if x.strip()]

    if len(vals) != INPUT_DIM:
        raise ValueError(
            f"Attesi {INPUT_DIM} valori in {txt_path}, trovati {len(vals)}"
        )

    return np.array(vals, dtype=np.float64)

# ============================================================
# WRAPPER COMPATIBILE COL TUO BFGS
# ============================================================

class SurrogateObjective:
    """
    Il solver ottimizza una variabile libera u.
    """

    def __init__(self, bundle, x_ref, device="cpu", delta_max=5.0, bound=300.0, clamp_f4_to_zero=True):
        self.bundle = bundle
        self.device = device
        self.delta_max = float(delta_max)
        self.bound = float(bound)
        self.clamp_f4_to_zero = bool(clamp_f4_to_zero)

        self.x_ref_np = np.array(x_ref, dtype=np.float64).copy()
        self.x_ref_t = torch.tensor(x_ref, dtype=torch.float32, device=device)

    # ----------------------------
    # trasformazioni u -> z -> x
    # ----------------------------
    def u_to_z_numpy(self, u_numpy: np.ndarray) -> np.ndarray:
        return self.x_ref_np + self.delta_max * np.tanh(u_numpy)

    def z_to_x_numpy(self, z_numpy: np.ndarray) -> np.ndarray:
        return np.clip(z_numpy, -self.bound, self.bound)

    def u_to_x_numpy(self, u_numpy: np.ndarray) -> np.ndarray:
        z = self.u_to_z_numpy(u_numpy)
        return self.z_to_x_numpy(z)

    def u_to_x_torch(self, u_tensor: torch.Tensor) -> torch.Tensor:
        z = self.x_ref_t + self.delta_max * torch.tanh(u_tensor)
        return torch.clamp(z, min=-self.bound, max=self.bound)

    # ----------------------------
    # valore dell'obiettivo totale
    # ----------------------------
    def f(self, u_numpy: np.ndarray) -> float:
        x_flat = self.u_to_x_numpy(u_numpy)
        x_mat = x_flat.reshape(M, 2)

        # parte nota via functions/obj.py
        val_known = f1(x_mat) + f2(x_mat) + f3(x_mat)

        # parte f4 via bundle torch
        x_input = torch.tensor(x_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        val_f4 = self.bundle.predict_f4(x_input).squeeze()
        if self.clamp_f4_to_zero:
            val_f4 = torch.clamp(val_f4, min=0.0)

        return float(val_known + val_f4.detach().cpu().item())

    # ----------------------------
    # gradiente totale rispetto a u
    # ----------------------------
    def grad_f(self, u_numpy: np.ndarray) -> np.ndarray:
        # =====================================
        # 1) gradiente della parte nota rispetto a x
        # =====================================
        x_flat = self.u_to_x_numpy(u_numpy)
        x_mat = x_flat.reshape(M, 2)

        grad_x_known = grad_total(x_mat).astype(np.float64)

        # =====================================
        # 2) chain rule da x a u
        # =====================================
        # z = x_ref + delta_max * tanh(u)
        # x = clip(z, -bound, bound)
        # dx/du = delta_max * (1 - tanh(u)^2) * I[-bound < z < bound]
        z_numpy = self.u_to_z_numpy(u_numpy)
        tanh_u = np.tanh(u_numpy)
        dz_du = self.delta_max * (1.0 - tanh_u**2)

        inside_bounds = ((z_numpy > -self.bound) & (z_numpy < self.bound)).astype(np.float64)
        dx_du = dz_du * inside_bounds

        grad_u_known = grad_x_known * dx_du

        # =====================================
        # 3) gradiente della parte f4 rispetto a u con autograd Torch
        # =====================================
        u_t = torch.tensor(
            u_numpy,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        x_flat_t = self.u_to_x_torch(u_t)
        x_input_t = x_flat_t.unsqueeze(0)

        val_f4 = self.bundle.predict_f4(x_input_t).squeeze()
        if self.clamp_f4_to_zero:
            val_f4 = torch.clamp(val_f4, min=0.0)

        val_f4.backward()
        grad_u_f4 = u_t.grad.detach().cpu().numpy().astype(np.float64)

        # =====================================
        # 4) gradiente totale
        # =====================================
        return grad_u_known + grad_u_f4

# ============================================================
# MAIN
# ============================================================

def find_solution():
    # Ripetibilità
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    print(f"[DEBUG] DEVICE = {DEVICE}")
    print(f"[DEBUG] CURRENT_DIR = {CURRENT_DIR}")
    print(f"[DEBUG] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[DEBUG] BUNDLE_DIR = {BUNDLE_DIR}")
    print(f"[DEBUG] X_REF_TXT_PATH = {X_REF_TXT_PATH}")
    print(f"[DEBUG] DELTA_MAX = {DELTA_MAX}")
    print(f"[DEBUG] SEED = {SEED}")
    print(f"[DEBUG] CLAMP_F4_TO_ZERO = {CLAMP_F4_TO_ZERO}")

    if not BUNDLE_DIR.exists():
        raise FileNotFoundError(f"Bundle directory non trovata: {BUNDLE_DIR}")

    if not X_REF_TXT_PATH.exists():
        raise FileNotFoundError(f"File x_ref non trovato: {X_REF_TXT_PATH}")

    bundle, config = load_bundle(BUNDLE_DIR, device=DEVICE)
    print("Config modello caricato:", config)

    x_ref = load_reference_from_txt(X_REF_TXT_PATH)
    print(f"Punto iniziale fissato da file: {X_REF_TXT_PATH}")

    objective = SurrogateObjective(
        bundle=bundle,
        x_ref=x_ref,
        device=DEVICE,
        delta_max=DELTA_MAX,
        bound=BOUND,
        clamp_f4_to_zero=CLAMP_F4_TO_ZERO
    )

    # u0 = 0 => x = x_ref
    u0 = np.zeros(INPUT_DIM, dtype=np.float64)

    print("Avvio GDM sulla funzione totale (localmente attorno a x_ref)...")
    
    u_star, history = gradient_descent_momentum(
        f=objective.f,
        grad_f=objective.grad_f,
        x0=u0,
        alpha=1,
        beta=0.3,
        tol=1e-5,
        max_iters=400
    )
    # Implementazione precedente con L-BFGS
    '''
    u_star = lbfgs_optimizer(
        f=objective.f,
        grad_f=objective.grad_f,
        x0=u0,
        m=10,
        tol=1e-5,
        max_iter=200
    )
'''
    u_star_t = torch.tensor(u_star, dtype=torch.float32, device=DEVICE)
    x_star = objective.u_to_x_torch(u_star_t).detach().cpu().numpy()

    f_star = objective.f(u_star)

    
    # Salva u*

    np.savetxt(CURRENT_DIR / "u_star.txt", u_star, fmt="%.15f")
    
    # Salva x*

    np.savetxt(CURRENT_DIR / "x_star.txt", x_star, fmt="%.15f")
    np.savetxt(CURRENT_DIR / "optimization_history.txt", history, fmt="%.15f")
    print(f"[INFO] History salvata in: {CURRENT_DIR / 'optimization_history.txt'}")

    print("[INFO] Ottimizzazione completata.")
    print(f"[INFO] Valore finale totale surrogato = {f_star:.6e}")
    
    print(f"[INFO] u* salvato in: {CURRENT_DIR / 'lbfgs_u_star.txt'}")
    print(f"[INFO] x* salvato in: {CURRENT_DIR / 'lbfgs_x_star.txt'}")


if __name__ == "__main__":
    find_solution()
