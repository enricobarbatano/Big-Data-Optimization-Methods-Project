import torch
from torch.optim import Optimizer

class CustomAdam(Optimizer):
    """
    Implementazione manuale di Adam.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Learning rate non valido: {lr}")
        
        # alpha, beta1, beta2, epsilon
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CustomAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g_k = p.grad
                state = self.state[p]
                
                alpha = group['lr']
                beta1, beta2 = group['betas']
                epsilon = group['eps']

                # --- Inizializzazione (m_0 = 0, v_0 = 0) ---
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m_k, v_k = state['m'], state['v']

                # k+1
                state['step'] += 1
                k_plus_1 = state['step']

                # 1. m_{k+1} = \beta_1 * m_k + (1 - \beta_1) * g_k
                m_k.mul_(beta1).add_(g_k, alpha=1 - beta1)

                # 2. v_{k+1} = \beta_2 * v_k + (1 - \beta_2) * g_k^2
                v_k.mul_(beta2).addcmul_(g_k, g_k, value=1 - beta2)

                # 3. Correzione del bias per m:
                m_hat = m_k / (1 - beta1 ** k_plus_1)

                # 4. Correzione del bias per v:
                v_hat = v_k / (1 - beta2 ** k_plus_1)

                # 5. Aggiornamento dei pesi
                denom = v_hat.sqrt() + epsilon
                
                # p = p - alpha * (m_hat / denom)
                p.addcdiv_(m_hat, denom, value=-alpha)

        return loss