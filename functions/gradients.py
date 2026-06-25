import numpy as np
from scipy.spatial.distance import pdist

r=100.0

# -------------------------------------------------------------
# f1
# -------------------------------------------------------------

def grad_f1(X):

    differenze = X[:, np.newaxis, :] - X[np.newaxis, :, :]

    distanze_quadrate = np.sum(differenze**2, axis=2)

    esponente = np.exp(-distanze_quadrate)[:, :, np.newaxis] * differenze

    grad = 0.2 * esponente

    return np.sum(grad, axis=1)

# -------------------------------------------------------------
# f2
# -------------------------------------------------------------

def grad_f2(X):

    return (1/100) * X
    
# -------------------------------------------------------------
# f3
# -------------------------------------------------------------

def grad_f3(X):
    
    m = X.shape[0] 
    
    I = np.arange(1, m + 1)
    
    valori_s = ((-1.0)**I) * I * 0.2
    
    S = np.column_stack((valori_s, valori_s))
    
    dist_quad = np.sum((X - S)**2, axis=1)

    c = np.log(r**2 + 1)

    k = 1 + dist_quad

    term1_1 = 4000 / (c * k)

    term1_2 = (np.log(k) / c) - 1

    term1 = term1_1 * term1_2

    term2 = X - S

    return term1[:, np.newaxis] * term2

# -------------------------------------------------------------
# Total
# -------------------------------------------------------------

def grad_total(X):
    x = X.reshape(1000, 2) 
    grad = grad_f1(x) + grad_f2(x) + grad_f3(x)
    return grad.flatten()
# -------------------------------------------------------------
# Verifica
# -------------------------------------------------------------

def verifica_gradiente(f, grad_f, X, epsilon=1e-5, tolleranza=1e-4):
    """
    Verifica il gradiente usando le differenze finite centrali.
    """
    gradiente_analitico = grad_f(X)
    gradiente_numerico = np.zeros_like(X)
    
    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        valore_originale = X[idx]
        
        X[idx] = valore_originale + epsilon
        f_avanti = f(X)
        
        X[idx] = valore_originale - epsilon
        f_indietro = f(X)
        
        X[idx] = valore_originale
        
        # (f(x + e) - f(x - e)) / (2*e)
        derivata_numerica = (f_avanti - f_indietro) / (2 * epsilon)
        gradiente_numerico[idx] = derivata_numerica
        
        it.iternext()
        
    # Calcoliamo la differenza (errore) tra i due gradienti
    errore_massimo = np.max(np.abs(gradiente_analitico - gradiente_numerico))
    
    print("--- Risultati del Test del Gradiente ---")
    print(f"Errore massimo assoluto: {errore_massimo:.2e}")
    
    if errore_massimo < tolleranza:
        print(" Gradiente corretto")
    else:
        print(" Gradiente errato.")
        print("\nPrimi 5 elementi del gradiente Analitico:\n", gradiente_analitico[:5])
        print("\nPrimi 5 elementi del gradiente Numerico:\n", gradiente_numerico[:5])