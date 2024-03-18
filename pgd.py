import numpy as np
from scipy import linalg, stats


### Toy Gaussian model

# EM
def em_tg(y, K, th):
    """Expectation Maximization Algorithm. Returns parameter estimates."""
    for k in range(K):
        th = np.append(th, th[k]/2 + y.mean()/2)  # Update parameter estimate.
    return th
    
# PGD
def pgd_tg(y, h, K, N, th, X):
    """Particle Gradient Ascent Algorithm. Returns parameter estimates."""
    D = y.size  # Extract dimension of latent variables.
    for k in range(K):
        # Update parameter estimate:
        th = np.append(th, th[k] + h*ave_grad_th_tg(th[k], X))
        # Update particle cloud:
        X = (X + h*grad_x_tg(y, th[k], X)
               + np.sqrt(2*h)*np.random.normal(0, 1, (D, N)))
    return th, X
    
def ave_grad_th_tg(th, X):
    """Returns theta-gradient of log density averaged over particles."""
    return X[:, 0].size*(X.mean() - th)

def grad_x_tg(y, th, X):
    """Returns x-gradient of log density vectorized over particles."""
    return (y + th - 2*X)

# IPLA
def ipla_tg(y, h, K, N, th, X):
    """Particle Gradient Ascent Algorithm. Returns parameter estimates."""
    D = y.size  # Extract dimension of latent variables.
    for k in range(K):
        # Update parameter estimate:
        th = np.append(th, th[k] + h*ave_grad_th_tg(th[k], X)+ np.sqrt(2*h/N)*np.random.normal(0, 1))
        # Update particle cloud:
        X = (X + h*grad_x_tg(y, th[k], X)
               + np.sqrt(2*h)*np.random.normal(0, 1, (D, N)))
    return th, X


### Bayesian logistic regression
# PGD
def pgd_blr(l, f, h, K, N, th, X, sigma):
    D = f[0, :].size  # Extract latent variable dimension.
    for k in range(K):
        Xk = X[:, -N:]  # Extract current particle cloud.
        thk = th[:, -1:]
        #Update particle cloud:
        Xkp1 = (Xk + h*grad_x_blr(thk, Xk, l, f, sigma)
                   + np.sqrt(2*h)*np.random.normal(0, 1, (D, N)))
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thk + h*ave_grad_th_blr(thk, Xk, sigma), axis = 1)  # Update theta.
    return th, X
# IPLA
def ipla_blr(l, f, h, K, N, th, X, sigma):
    D = f[0, :].size  # Extract latent variable dimension.
    for k in range(K):
        Xk = X[:, -N:]  # Extract current particle cloud.
        thk = th[:, -1:]
        #Update particle cloud:
        Xkp1 = (Xk + h*grad_x_blr(thk, Xk, l, f, sigma)
                   + np.sqrt(2*h)*np.random.normal(0, 1, (D, N)))
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thk + h*ave_grad_th_blr(thk, Xk, sigma)
                       + np.sqrt(2*h/N)*np.random.normal(0, D), axis = 1)  # Update theta.
    return th, X

def ave_grad_th_blr(th, x, sigma):
    """Returns theta-gradient of log density averaged over particle cloud."""
    return (np.mean(x, axis = 1)-th)/sigma**2

def grad_x_blr(th, x, l, f, sigma):
    """Returns x-gradient of log density vectorized over particles."""
    s = 1/(1+np.exp(- np.matmul(f, x)))
    return np.matmul((l-s).transpose(), f).transpose() - (x-th)/sigma**2
