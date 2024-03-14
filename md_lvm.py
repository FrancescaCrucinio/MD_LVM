import numpy as np
from scipy import linalg, stats

# modules from particles
import particles  # core module
from particles import smc_samplers as ssp
from particles import resampling as rs

# General functions

def rwm_proposal(v, W):
    arr = ssp.view_2d_array(v)
    N, d = arr.shape
    m, cov = rs.wmean_and_cov(W, arr)
    scale = 2.38 / np.sqrt(d)
    L = scale * linalg.cholesky(cov, lower=True)
    arr_prop = arr + stats.norm.rvs(size=arr.shape) @ L.T
    return arr_prop

# Toy Gaussian model

def ll_toy_lvm(theta, x, data):
    """Returns -U"""
    log_p = -0.5*np.sum((x - theta)**2, axis = 1)
    log_p = log_p - 0.5*np.sum((x-data.T)**2, axis = 1)
    return log_p

    
def rwm_accept_toy_lvm_varying_gamma(v, prop, theta_seq, gamma_seq, data):
    n = theta_seq.size
    log_acceptance = 0.5*np.prod(1-gamma_seq)*np.sum((v - prop)**2, axis = 1)
    for k in range(n):
        log_acceptance = log_acceptance + gamma_seq[k]*np.prod(1-gamma_seq[(k+2):])*(ll_toy_lvm(theta_seq[k], prop, data) - ll_toy_lvm(theta_seq[k], v, data))
    accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
    output = ssp.view_2d_array(v)
    output[accepted, :] = prop[accepted, :]
    return output

def rwm_accept_toy_lvm(v, prop, theta_seq, gamma, data):
    n = theta_seq.size
    log_acceptance = 0.5*((1-gamma)**n)*np.sum((v - prop)**2, axis = 1)
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


def md_toy_lvm_varying_gamma(y, gamma, Niter, N, th0, X0):
    D = X0.shape[1]
    x = np.zeros((Niter, N, D))
    theta = np.zeros(Niter)
    theta[0] = th0
    x[0, :, :] = X0
    W = np.ones(N)/N
    for n in range(1, Niter):
        theta[n] = theta[n-1] + gamma[n]*(np.sum(np.sum(x[n-1, :, :], axis = 1)*W)-D*theta[n-1])
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :, :] = x[n-1, ancestors, :]
        # MCMC move
        prop = rwm_proposal(x[n-1, :, :], W)
        x[n, :, :] = rwm_accept_toy_lvm_varying_gamma(x[n-1, :, :], prop, theta[:n], gamma[1:(n+1)], y)
        # reweight
        logW = ll_toy_lvm(theta[n], x[n, :, :], y) + 0.5*np.prod(1-gamma[1:n])*np.sum(x[n, :, :]**2, axis = 1)
        for k in range(n-1):
            logW = logW - gamma[k+1]*np.prod(1-gamma[(k+2):n])*ll_toy_lvm(theta[k], x[n, :, :], y)
        logW = gamma[n]*logW
        W = rs.exp_and_normalise(logW)
    return theta, x, W

def md_toy_lvm_adaptive(y, N, th0, X0, epsilon, Niter = 1000):
    D = X0.shape[1]
    gamma = np.array([0])
    Cn = np.array([1])
    theta = np.array([])
    n = -1
    while((Cn[n] > epsilon) & (n < Niter)):
        n = n+1
        if (n == 0):
            theta = np.append(theta, th0)
            x = np.random.normal(size = (N, D))
        else:
            Cn = np.append(Cn, np.prod(1-gamma[1:(n+1)]))
            theta[n] = theta[n-1] + gamma[n-1]*(np.sum(np.sum(x, axis = 1)*W)-D*theta[n-1])
            # resample
            ancestors = rs.resampling('stratified', W)
            x = x[ancestors, :]
            # MCMC move
            prop = rwm_proposal(x, W)
            x[n, :, :] = rwm_accept_toy_lvm(x, prop, theta[:n], gamma[:n], y)
        # reweight
        logW = ll_toy_lvm(theta[n], x, y) + 0.5*np.prod(1-gamma[:n])*np.sum(x**2, axis = 1)
        for k in range(n):
            logW = logW - gamma[k]*np.prod(1-gamma[k:n])*ll_toy_lvm(theta[k], x, y)
        new_l = ssp.next_annealing_epn(gamma[n], 0.5, logW)
        gamma = np.append(gamma, new_l - gamma[n])
        logW = gamma[n]*logW
        W = rs.exp_and_normalise(logW)
    return theta, x, W