import numpy as np
from scipy import linalg, stats
from particles import resampling as rs

def smc_mml_multimodal(N, Niter, data):
    theta = np.random.uniform(-50, 50, size = N)
    theta_estimate = np.zeros(Niter)
    theta_estimate[0] = np.mean(theta)
    logW = np.zeros(N)
    mu = np.zeros(N)
    sigma = np.zeros(N)
    for n in range(1, Niter):
        incremental_w = np.zeros(N)
        for i in range(4):
            incremental_w = incremental_w - 0.525*np.log(0.05+(data[i] - theta)**2)
        logW = logW + incremental_w
        W = rs.exp_and_normalise(logW)
        if(rs.essl(logW)<N/2):
            ancestors = rs.resampling('stratified', W)
            theta = theta[ancestors]
            logW = np.zeros(N)
        z = np.zeros((N, 4))
        for k in range(N):
            for i in range(4):
                z[k, i] = np.random.gamma(shape = 0.525, scale =1/(0.025+(data[i]-theta[k])**2/2), size = 1)
            if(n>1):
                sigma_old = sigma[k]
                mu_old = mu[k]
                sigma[k] = 1/(1/sigma_old+np.sum(z[k, :]))
                mu[k] = sigma[k]*(mu_old/sigma_old+np.sum(data*z[k, :]))
            else:
                sigma[k] = 1/np.sum(z[k, :])
                mu[k] = sigma[k]*np.sum(data*z[k, :])
            theta[k] = np.random.normal(loc = mu[k], scale = np.sqrt(sigma[k]), size = 1)
        theta_estimate[n] = np.sum(theta*W)
    return theta_estimate, z, W