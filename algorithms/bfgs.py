import numpy as np

def two_loop_recursion(g_k, S, Y):
    q = np.copy(g_k)
    alphas = {}
    rhos = {}
    m_current = len(S)
    
    # 1. Backward Loop
    for i in reversed(range(m_current)):
        s_i = S[i]
        y_i = Y[i]
        
        ys = np.dot(y_i, s_i)
        if abs(ys) <= 1e-12:
            continue

        rho_i = 1.0 / np.dot(y_i, s_i)
        rhos[i] = rho_i
        
        alpha_i = rho_i * np.dot(s_i, q)
        alphas[i] = alpha_i
        
        q -= alpha_i * y_i
        
    # H0 = gamma * I
    if m_current > 0:
        yy = np.dot(Y[-1], Y[-1])
        if abs(yy) > 1e-12:
            gamma = np.dot(S[-1], Y[-1]) / yy
        else:
            gamma = 1.0
        r = gamma * q
    else:
        r = q
        
    # 2. Forward Loop 2
    for i in range(m_current):
        if i not in rhos or i not in alphas:
            continue

        s_i = S[i]
        y_i = Y[i]

        beta = rhos[i] * np.dot(y_i, r)
        r += s_i * (alphas[i] - beta)
    return r

def lbfgs_optimizer(f, grad_f, x0, m=10, tol=1e-5, max_iter=1000):
    """
    Metodo L-BFGS.
    """
    x = x0.flatten()
    S = []
    Y = []
    
    g = grad_f(x)
    
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            print(f"Ottimo trovato all'iterazione {k}")
            break
            
        # Calcolo di d
        r = two_loop_recursion(g, S, Y)
        d = -r
        
        # Line Search Inesatta
        alpha = 1.0
        gamma_armijo = 1e-4
        delta_armijo = 0.5
        
        # Condizione di Armijo
        while f(x + alpha * d) > f(x) + gamma_armijo * alpha * np.dot(g, d):
            alpha *= delta_armijo
            
        x_next = x + alpha * d
        g_next = grad_f(x_next)
        
        s_k = x_next - x
        y_k = g_next - g
        
        # Salvataggio in memoria
        ys = np.dot(y_k, s_k)
        if ys > 1e-12:
            S.append(s_k)
            Y.append(y_k)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)

            
        x = x_next
        g = g_next

        print(f"Iterazione {k+1}: f(x) = {f(x)}, ||grad|| = {np.linalg.norm(g)}")
        
    return x