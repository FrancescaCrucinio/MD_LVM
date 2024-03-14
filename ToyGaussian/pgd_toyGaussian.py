import numpy as np
from scipy import linalg, stats

# EM
def em(y, K, th):
    """Expectation Maximization Algorithm. Returns parameter estimates."""
    for k in range(K):
        th = np.append(th, th[k]/2 + y.mean()/2)  # Update parameter estimate.
    return th
    
# PGD
def pgd(y, h, K, N, th, X):
    """Particle Gradient Ascent Algorithm. Returns parameter estimates."""
    D = y.size  # Extract dimension of latent variables.
    for k in range(K):
        # Update parameter estimate:
        th = np.append(th, th[k] + h*ave_grad_th(th[k], X))
        # Update particle cloud:
        X = (X + h*grad_x(y, th[k], X)
               + np.sqrt(2*h)*np.random.normal(0, 1, (D, N)))
    return th, X
    
def ave_grad_th(th, X):
    """Returns theta-gradient of log density averaged over particles."""
    return X[:, 0].size*(theta_opt(X) - th)

def grad_x(y, th, X):
    """Returns x-gradient of log density vectorized over particles."""
    return (y + th - 2*X)

def theta_opt(X):
    return X.mean()  # Return optimal parameter for particle cloud X.

# IPLA
def ipla(y, h, K, N, th, X):
    """Particle Gradient Ascent Algorithm. Returns parameter estimates."""
    D = y.size  # Extract dimension of latent variables.
    for k in range(K):
        # Update parameter estimate:
        th = np.append(th, th[k] + h*ave_grad_th(th[k], X)+ np.sqrt(2*h/N)*np.random.normal(0, 1))
        # Update particle cloud:
        X = (X + h*grad_x(y, th[k], X)
               + np.sqrt(2*h)*np.random.normal(0, 1, (D, N)))
    return th, X
