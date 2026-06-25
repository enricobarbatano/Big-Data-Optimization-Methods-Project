import numpy as np

def gradient_descent_momentum(f, grad_f, x0, alpha, beta, max_iters=1000, tol=1e-6):
    """
    Metodo del gradiente con momentum (heavy-ball).
    """
    # Inizializzazione
    x = np.array(x0, dtype=float)
    x_prev = np.copy(x) 
    
    history = [f(x)]
    
    for k in range(max_iters):
        g = grad_f(x)
        
        # Criterio di arresto
        if np.linalg.norm(g) < tol:
            print(f"Convergenza raggiunta all'iterazione {k}")
            break
            
        # Aggiornamento con la formula Heavy-Ball:
        x_next = x - alpha * g + beta * (x - x_prev)
        
        x_prev = np.copy(x)
        x = np.copy(x_next)
        
        current_value = f(x)
        current_grad = np.linalg.norm(g)
        history.append(current_value)

        print(f"Iterazione {k+1}: ||grad|| = {current_grad:.6e}, Value = {current_value:.6e}")
        
    else:
        print(f"Raggiunto il numero massimo di iterazioni ({max_iters}) senza convergere strettamente.")
        
    return x, history