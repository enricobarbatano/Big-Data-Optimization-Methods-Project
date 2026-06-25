import subprocess
import numpy as np
from scipy.spatial.distance import pdist
# definzione della funzione obiettivo sulla base dei costi f1, f2, f3  

# dimensione = 1000
m=1000

r=100.0


# -------------------------------------------------------------
# f1
# -------------------------------------------------------------

def f1(x):

    # pdist calcola le distanze per tutte le coppie non ordinate (i < j).
    # 'sqeuclidean' calcola direttamente il quadrato della distanza euclidea: ||xi - xj||^2
    distanze_quadrate = pdist(x, metric='sqeuclidean')
    
    dispersioni = 0.1 * (1 - np.exp(-distanze_quadrate))
    
    f1 = np.sum(dispersioni)
    return f1

# -------------------------------------------------------------
# f2
# -------------------------------------------------------------

def f2(x):
    norma = np.sum(x**2, axis=1) # norma = prodotto scalare ** 2
    aux = (1/m) * (5*norma + 80000)
    f2 = np.sum(aux)
    return f2

# -------------------------------------------------------------
# f3
# -------------------------------------------------------------

def f3(X):

    m = X.shape[0] 
    
    I = np.arange(1, m + 1)
    
    # s_i = (-1)^i * i * 0.2
    valori_s = ((-1.0)**I) * I * 0.2
    
    S = np.column_stack((valori_s, valori_s))
    
    dist_quad = np.sum((X - S)**2, axis=1)
    
    r = 100.0
    denominatore = np.log(1 + r**2)
    
    deviazioni = 1000 * ((np.log(1 + dist_quad) / denominatore) - 1.0)**2
    
    return np.sum(deviazioni)

# -------------------------------------------------------------
# f4
# -------------------------------------------------------------

def f4():
    # Chiama l'eseguibile esterno e cattura l'output
    result =subprocess.run([r'.\executables\mobd.exe', r'.\executables\x.txt', '-b'], capture_output=True, text=True)
    return result.stdout
    
# -------------------------------------------------------------
# f_total
# -------------------------------------------------------------

def f_total(X):
    x = X.reshape(1000, 2) 
    return f1(x) + f2(x) + f3(x)