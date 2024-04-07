import numpy as np
from scipy import linalg, stats, optimize
from scipy.stats import logistic

# modules from particles
import particles  # core module
from particles import smc_samplers as ssp
from particles import resampling as rs

### General functions

def rwm_proposal(v, W):
    arr = ssp.view_2d_array(v)
    N, d = arr.shape
    m, cov = rs.wmean_and_cov(W, arr)
    scale = 2.38 / np.sqrt(d)
    L = scale * linalg.cholesky(cov, lower=True)
    arr_prop = arr + stats.norm.rvs(size=arr.shape) @ L.T
    return arr_prop

### Toy Gaussian model

# log-likelihood
def ll_toy_lvm(theta, x, data):
    """Returns -U"""
    log_p = -0.5*np.sum((x - theta)**2, axis = 1)
    log_p = log_p - 0.5*np.sum((x-data.T)**2, axis = 1)
    return log_p
## MD
# accept/reject step
def rwm_accept_toy_lvm(v, prop, theta_seq, gamma, data):
    n = theta_seq.size
    log_acceptance = 0.5*((1-gamma)**n)*np.sum(v**2 - prop**2, axis = 1)
    for k in range(n):
        log_acceptance = log_acceptance + gamma*(1-gamma)**(n-k-2)*(ll_toy_lvm(theta_seq[k], prop, data) - ll_toy_lvm(theta_seq[k], v, data))
    accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
    output = ssp.view_2d_array(v)
    output[accepted, :] = prop[accepted, :]
    return output

def md_toy_lvm(y, gamma, Niter, N, th0, X0):
    D = X0.shape[1]
    x = np.zeros((Niter, N, D))
    theta = np.zeros(Niter)
    theta[0] = th0
    x[0, :, :] = X0
    W = np.ones(N)/N
    for n in range(1, Niter):
        theta[n] = theta[n-1] + gamma*(np.sum(np.sum(x[n-1, :, :], axis = 1)*W)-D*theta[n-1])
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :, :] = x[n-1, ancestors, :]
        # MCMC move
        prop = rwm_proposal(x[n-1, :, :], W)
        x[n, :, :] = rwm_accept_toy_lvm(x[n-1, :, :], prop, theta[:n], gamma, y)
        # reweight
        logW = ll_toy_lvm(theta[n], x[n, :, :], y) + 0.5*((1-gamma)**(n-1))*np.sum(x[n, :, :]**2, axis = 1)
        for k in range(n-1):
            logW = logW - gamma*((1-gamma)**(n-k-2))*ll_toy_lvm(theta[k], x[n, :, :], y)
        logW = gamma*logW
        W = rs.exp_and_normalise(logW)
    return theta, x, W
## SMC
def rwm_accept_toy_lvm_fast(v, prop, theta_current, gamma, data, n):
    log_acceptance = 0.5*(1-gamma)**n*np.sum(v**2 - prop**2, axis = 1)+(1-(1-gamma)**n)*(ll_toy_lvm(theta_current, prop, data) - ll_toy_lvm(theta_current, v, data))
    accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
    output = ssp.view_2d_array(v)
    output[accepted, :] = prop[accepted, :]
    return output

def md_toy_lvm_fast(y, gamma, Niter, N, th0, X0):
    D = X0.shape[1]
    x = np.zeros((Niter, N, D))
    theta = np.zeros(Niter)
    theta[0] = th0
    x[0, :, :] = X0
    W = np.ones(N)/N
    for n in range(1, Niter):
        theta[n] = theta[n-1] + gamma*(np.sum(np.sum(x[n-1, :, :], axis = 1)*W)-D*theta[n-1])
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :, :] = x[n-1, ancestors, :]
        # MCMC move
        prop = rwm_proposal(x[n-1, :, :], W)
        x[n, :, :] = rwm_accept_toy_lvm_fast(x[n-1, :, :], prop, theta[n-1], gamma, y, n)
        # reweight  
        logW = (1-(1-gamma)**n)*ll_toy_lvm(theta[n-1], x[n, :, :], y) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_toy_lvm(theta[n-2], x[n, :, :], y) 
        W = rs.exp_and_normalise(logW)
    return theta, x, W

def md_toy_lvm_fast2(y, gamma, Niter, N, th0, X0):
    D = X0.shape[1]
    x = np.zeros((Niter, N, D))
    theta = np.zeros(Niter)
    theta[0] = th0
    x[0, :, :] = X0
    W = np.ones(N)/N
    for n in range(1, Niter):
        theta[n] = theta[n-1] + gamma*(np.sum(np.sum(x[n-1, :, :], axis = 1)*W)-D*theta[n-1])
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :, :] = x[n-1, ancestors, :]
        # MCMC move
        prop = rwm_proposal(x[n-1, :, :], W)
        x[n, :, :] = rwm_accept_toy_lvm_fast(x[n-1, :, :], prop, theta[n], gamma, y, n)
        # reweight  
        logW = (1-(1-gamma)**n)*ll_toy_lvm(theta[n], x[n, :, :], y) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_toy_lvm(theta[n-1], x[n, :, :], y) 
        W = rs.exp_and_normalise(logW)
    return theta, x, W

### Bayesian logistic regression
# log-likelihood
def ll_bayesian_lr(th, x, l, f, sigma):
    s = 1/(1+np.exp(- np.matmul(f, x.T)))
    return np.matmul(l, np.log(s))+np.matmul(1-l, np.log(1-s)) - 0.5*np.sum(((x-th).T)**2, axis = 0)/sigma**2
# accept/reject step
def rwm_accept_bayesian_lr_fast(v, prop, theta_current, gamma, data, f, sigma, n):
    log_acceptance = 0.5*(1-gamma)**n*np.sum(v**2 - prop**2, axis = 1)+(1-(1-gamma)**n)*(ll_bayesian_lr(theta_current, prop, data, f, sigma) - ll_bayesian_lr(theta_current, v, data, f, sigma))
    accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
    output = ssp.view_2d_array(v)
    output[accepted, :] = prop[accepted, :]
    return output
# SMC
def md_toy_bayesian_lr_fast(y, gamma, Niter, N, th0, X0, f, sigma):
    D = X0.shape[1]
    x = np.zeros((Niter, N, D))
    theta = np.zeros((Niter, D))
    theta[0, :] = th0
    x[0, :, :] = X0
    W = np.ones(N)/N
    for n in range(1, Niter):
        theta[n, :] = theta[n-1, :] + gamma*(np.sum(x[n-1, :, :].T*W, axis = 1)-theta[n-1, :])/sigma**2
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :, :] = x[n-1, ancestors, :]
        # MCMC move
        prop = rwm_proposal(x[n-1, :, :], W)
        x[n, :, :] = rwm_accept_bayesian_lr_fast(x[n-1, :, :], prop, theta[n-1, :], gamma, y, f, sigma, n)
        # reweight
        logW = (1-(1-gamma)**n)*ll_bayesian_lr(theta[n-1, :], x[n, :, :], y, f, sigma) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_bayesian_lr(theta[n-2, :], x[n, :, :], y, f, sigma)
        W = rs.exp_and_normalise(logW)
    return theta, x, W

def md_toy_bayesian_lr_fast2(y, gamma, Niter, N, th0, X0, f, sigma):
    D = X0.shape[1]
    x = np.zeros((Niter, N, D))
    theta = np.zeros((Niter, D))
    theta[0, :] = th0
    x[0, :, :] = X0
    W = np.ones(N)/N
    for n in range(1, Niter):
        theta[n, :] = theta[n-1, :] + gamma*(np.sum(x[n-1, :, :].T*W, axis = 1)-theta[n-1, :])/sigma**2
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :, :] = x[n-1, ancestors, :]
        # MCMC move
        prop = rwm_proposal(x[n-1, :, :], W)
        x[n, :, :] = rwm_accept_bayesian_lr_fast(x[n-1, :, :], prop, theta[n, :], gamma, y, f, sigma, n)
        # reweight
        logW = (1-(1-gamma)**n)*ll_bayesian_lr(theta[n, :], x[n, :, :], y, f, sigma) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_bayesian_lr(theta[n-1, :], x[n, :, :], y, f, sigma)
        W = rs.exp_and_normalise(logW)
    return theta, x, W





### Independent component analysis
# log-likelihood
def ll_ica(sigma, A, x, alpha, y):
    d = y.shape[0]
    mixture = alpha*logistic.pdf(x, scale = 1/2) + (1-alpha)*(x == 0)
    prior = np.sum(np.log(mixture), axis = 1)
    likelihood = -0.5*d*np.log(2*np.pi*sigma**2)-0.5*np.sum(np.subtract(np.matmul(A, x.T), y)**2, axis = 0)/sigma**2
    return prior+likelihood
# accept/reject step
def rwm_accept_ica_fast(v, prop, sigma_current, A_current, alpha, gamma, data, n):
    log_acceptance = 0.5*(1-gamma)**n*np.sum(v**2 - prop**2, axis = 1)+(1-(1-gamma)**n)*(ll_ica(sigma_current, A_current, prop, alpha, data) - ll_ica(sigma_current, A_current, v, alpha, data))
    accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
    output = ssp.view_2d_array(v)
    output[accepted, :] = prop[accepted, :]
    return output
# gradients
def ica_gradient_sigma(A, sigma, Y, x):
    d, m = Y.shape
    log_gradient = np.sum((Y - np.matmul(A, x.T))**2, axis = 0)/sigma**3 - d/sigma
    return log_gradient

def ica_gradient_A(A, sigma, Y, x):
    d, m = Y.shape
    p = x.shape[1]
    gradient = np.zeros((m, d, p))
    for i in range(m):
        tmp = np.outer(Y[:, i] - np.matmul(A, x.T)[:, i], x[i, :])
        gradient[i, :, :] = np.sign(tmp) * np.exp(np.log(np.abs(tmp)) - np.log(sigma**2))
    return gradient

# SMC
def md_ica_fast(y, gamma, Niter, N, A0, sigma0, X0, alpha):
    d = y.shape[0]
    p = X0.shape[1]
    x = np.zeros((Niter, N, p))
    
    A = np.zeros((Niter, d, p))
    sigma = np.zeros((Niter))

    A[0,:,:] = A0
    sigma[0] = sigma0

    x[0, :, :] = X0
    W = np.ones(N)/N
    for n in range(1, Niter):
        sigma[n] = sigma[n-1] + gamma*np.sum(ica_gradient_sigma(A[n-1, :, :], sigma[n-1], y, x[n-1, :, :])*W)
        A[n, :, :] = A[n-1, :, :] + gamma*np.sum(ica_gradient_A(A[n-1, :, :], sigma[n-1], y, x[n-1, :, :]).T*W, axis = 2).T
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :, :] = x[n-1, ancestors, :]
        # MCMC move
        prop = rwm_proposal(x[n-1, :, :], W)
        x[n, :, :] = rwm_accept_ica_fast(x[n-1, :, :], prop, sigma[n-1], A[n-1, :, :], alpha, gamma, y, n)
        # reweight
        logW = (1-(1-gamma)**n)*ll_ica(sigma[n-1], A[n-1, :, :], x[n, :, :], alpha, y) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_ica(sigma[n-2], A[n-2, :, :], x[n, :, :], alpha, y) 
        W = rs.exp_and_normalise(logW)
    return A, sigma, x, W









# # def rwm_accept_toy_lvm_varying_gamma(v, prop, theta_seq, gamma_seq, data):
# #     n = theta_seq.size
# #     log_acceptance = 0.5*np.prod(1-gamma_seq)*np.sum(v**2 - prop**2, axis = 1)
# #     for k in range(n):
# #         log_acceptance = log_acceptance + gamma_seq[k]*np.prod(1-gamma_seq[(k+2):])*(ll_toy_lvm(theta_seq[k], prop, data) - ll_toy_lvm(theta_seq[k], v, data))
# #     accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
# #     output = ssp.view_2d_array(v)
# #     output[accepted, :] = prop[accepted, :]
# #     return output






# # def md_toy_lvm_varying_gamma(y, gamma, Niter, N, th0, X0):
# #     D = X0.shape[1]
# #     x = np.zeros((Niter, N, D))
# #     theta = np.zeros(Niter)
# #     theta[0] = th0
# #     x[0, :, :] = X0
# #     W = np.ones(N)/N
# #     for n in range(1, Niter):
#         theta[n] = theta[n-1] + gamma[n]*(np.sum(np.sum(x[n-1, :, :], axis = 1)*W)-D*theta[n-1])
#         if (n > 1):
#             # resample
#             ancestors = rs.resampling('stratified', W)
#             x[n-1, :, :] = x[n-1, ancestors, :]
#         # MCMC move
#         prop = rwm_proposal(x[n-1, :, :], W)
#         x[n, :, :] = rwm_accept_toy_lvm_varying_gamma(x[n-1, :, :], prop, theta[:n], gamma[1:(n+1)], y)
#         # reweight
#         logW = ll_toy_lvm(theta[n], x[n, :, :], y) + 0.5*np.prod(1-gamma[1:n])*np.sum(x[n, :, :]**2, axis = 1)
#         for k in range(n-1):
#             logW = logW - gamma[k+1]*np.prod(1-gamma[(k+2):n])*ll_toy_lvm(theta[k], x[n, :, :], y)
#         logW = gamma[n]*logW
#         W = rs.exp_and_normalise(logW)
#     return theta, x, W

# def next_annealing_epn_md(epn, alpha, lw_lambda, lw_old):
#     """Find next annealing exponent by solving ESS(exp(lw)) = alpha * N.

#     Parameters
#     ----------
#     epn: float
#         current exponent
#     alpha: float in (0, 1)
#         defines the ESS target
#     lw:  numpy array of shape (N,)
#         log-weights
#     """
#     N = lw_lambda.shape[0]
#     def f(e):
#         ess = rs.essl(e * lw_lambda + lw_old) if e > 0.0 else N  # avoid 0 x inf issue when e==0
#         return ess - alpha * N
#     if f(1. - epn) < 0.:
#         return epn + optimize.brentq(f, 0.0, 1.0 - epn)
#     else:
#         return 1.0


# def md_toy_lvm_adaptive(y, N, th0, X0, epsilon, Niter = 1000):
#     D = X0.shape[1]
#     gamma = np.array([0])
#     Cn = np.array([1])
#     theta = np.array([])
#     n = -1
#     lambda_old = 0
#     while((n < Niter)):
#         n = n+1
#         if (n == 0):
#             theta = np.append(theta, th0)
#             x = np.random.normal(size = (N, D))
#         else:
#             Cn = np.append(Cn, 1-lambda_old)
#             theta = np.append(theta, theta[n-1] + 0.001*(np.sum(np.sum(x, axis = 1)*W)-D*theta[n-1]))
#             # resample
#             ancestors = rs.resampling('stratified', W)
#             x = x[ancestors, :]
#             # MCMC move
#             prop = rwm_proposal(x, W)
#             x = rwm_accept_toy_lvm_fast(x, prop, theta[n-1], lambda_old, y, n)
#         # reweight
#         logW_fixed = -0.5*lambda_old*np.sum(x**2, axis = 1)
#         if(n>1):
#             logW_fixed = logW_fixed - lambda_old*ll_toy_lvm(theta[n-2], x, y) 
#         logW_lambda = 0.5*np.sum(x**2, axis = 1)+ll_toy_lvm(theta[n-1], x, y)
#         new_l = next_annealing_epn_md(lambda_old, 0.5, logW_lambda, logW_fixed)
#         logW = logW_fixed + new_l*logW_lambda
#         lambda_old = new_l
#         W = rs.exp_and_normalise(logW)
#     return theta, x, W