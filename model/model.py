import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

class LinearManual(nn.Module):
    """Layer fully-connected implementato a mano (no nn.Linear)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        limit = (6.0 / (in_features + out_features)) ** 0.5
        w = torch.empty(in_features, out_features, dtype=torch.float32).uniform_(-limit, limit)
        b = torch.zeros(out_features, dtype=torch.float32)
        self.W = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W + self.b


class GELUManual(nn.Module):
    """GELU fatta a mano: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715x^3)))."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.sqrt(torch.tensor(2.0 / np.pi, dtype=x.dtype, device=x.device))
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


class TanhManual(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class IdentityManual(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


_ACTS = {
    'gelu': GELUManual,
    'tanh': TanhManual,
    'identity': IdentityManual,
}


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name not in _ACTS:
        raise ValueError(f"Attivazione non supportata: {name}")
    return _ACTS[name]()


class F4SurrogateNet(nn.Module):
    """
    Rete custom per regressione di f4.
    """
    def __init__(self, input_dim: int = 2000, hidden_dims: List[int] = None,
                 output_dim: int = 1, activation: str = 'gelu'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 516, 256, 128, 64, 32, 16]
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.activation = activation

        layers = []
        for i in range(len(dims) - 1):
            layers.append(LinearManual(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_get_activation(activation))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class F4SurrogateBundle:
    """
    Bundle che contiene modello + statistiche di normalizzazione, da usare per ottimizzazione.
    """
    def __init__(self,
                 model: F4SurrogateNet,
                 x_mean: np.ndarray,
                 x_std: np.ndarray,
                 y_mean: np.ndarray,
                 y_std: np.ndarray,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.x_mean = torch.tensor(x_mean, dtype=torch.float32, device=device)
        self.x_std = torch.tensor(x_std, dtype=torch.float32, device=device)
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device)
        self.y_std = torch.tensor(y_std, dtype=torch.float32, device=device)

    def normalize_x(self, x_tensor: torch.Tensor) -> torch.Tensor:
        return (x_tensor - self.x_mean) / self.x_std

    def denormalize_y(self, y_tensor: torch.Tensor) -> torch.Tensor:
        return y_tensor * self.y_std + self.y_mean

    def predict_f4(self, x_tensor: torch.Tensor) -> torch.Tensor:
        x_norm = self.normalize_x(x_tensor)
        y_hat_norm = self.model(x_norm)
        y_hat_real = self.denormalize_y(y_hat_norm)
        return torch.nn.functional.softplus(y_hat_real)  


def save_bundle(model: F4SurrogateNet,
                outdir: Path,
                x_mean: np.ndarray,
                x_std: np.ndarray,
                y_mean: np.ndarray,
                y_std: np.ndarray,
                input_dim: int,
                hidden_dims: List[int],
                output_dim: int = 1,
                activation: str = 'gelu') -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), outdir / 'f4_surrogate_weights.pt')
    np.savez(outdir / 'f4_surrogate_stats.npz',
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    config: Dict[str, Any] = {
        'input_dim': input_dim,
        'hidden_dims': list(hidden_dims),
        'output_dim': output_dim,
        'activation': activation,
    }
    with open(outdir / 'f4_surrogate_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def load_bundle(outdir: Path, device: str = 'cpu') -> Tuple[F4SurrogateBundle, Dict[str, Any]]:
    outdir = Path(outdir)
    with open(outdir / 'f4_surrogate_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    model = F4SurrogateNet(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        output_dim=config['output_dim'],
        activation=config['activation'],
    )
    state_dict = torch.load(outdir / 'f4_surrogate_weights.pt', map_location=device)
    model.load_state_dict(state_dict)

    stats = np.load(outdir / 'f4_surrogate_stats.npz')
    bundle = F4SurrogateBundle(model=model,
                               x_mean=stats['x_mean'],
                               x_std=stats['x_std'],
                               y_mean=stats['y_mean'],
                               y_std=stats['y_std'],
                               device=device)
    return bundle, config
