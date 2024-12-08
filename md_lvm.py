import numpy as np
from scipy import linalg, stats, optimize
from scipy.stats import logistic
from sklearn.cluster import KMeans

# modules from particles
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
        x[n, :, :] = rwm_accept_toy_lvm(x[n-1, :, :], prop, theta[:(n-1)], gamma, y)
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
        x[n, :, :] = rwm_accept_toy_lvm_fast(x[n-1, :, :], prop, theta[n-2], gamma, y, n-1)
        # reweight  
        logW = (1-(1-gamma)**n)*ll_toy_lvm(theta[n-1], x[n, :, :], y) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_toy_lvm(theta[n-2], x[n, :, :], y) 
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
        x[n, :, :] = rwm_accept_bayesian_lr_fast(x[n-1, :, :], prop, theta[n-2, :], gamma, y, f, sigma, n-1)
        # reweight
        logW = (1-(1-gamma)**n)*ll_bayesian_lr(theta[n-1, :], x[n, :, :], y, f, sigma) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_bayesian_lr(theta[n-2, :], x[n, :, :], y, f, sigma)
        W = rs.exp_and_normalise(logW)
    return theta, x, W

def md_toy_bayesian_lr_fast_lookahead(y, gamma, Niter, N, th0, X0, f, sigma):
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
        logW = (1-(1-gamma)**n)*ll_bayesian_lr(theta[n, :], x[n, :, :], y, f, sigma) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_bayesian_lr(theta[n-1, :], x[n, :, :], y, f, sigma)
        W = rs.exp_and_normalise(logW)
    return theta, x, W

### Multimodal example
# log-likelihood
def ll_multimodal(theta, x, y):
    return -np.sum(0.475*np.log(x)+0.025*x+0.5*x*(y-theta)**2, axis = 1)
# accept/reject step
def rwm_accept_multimodal_fast(v, prop, theta_current, gamma, data, n):
    log_acceptance = 0.5*(1-gamma)**n*np.sum(v**2 - prop**2, axis = 1)+(1-(1-gamma)**n)*(ll_multimodal(theta_current, prop, data) - ll_multimodal(theta_current, v, data))
    accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
    ssp.view_2d_array(v)
    output = ssp.view_2d_array(v)
    output[accepted, :] = prop[accepted, :]
    return output
# SMC
def md_multimodal_fast(y, gamma, Niter, N, th0, X0):
    ndata = y.size
    x = np.zeros((Niter, N, ndata))
    theta = np.zeros(Niter)
    theta[0] = th0
    x[0, :, :] = X0
    W = np.ones(N)/N
    for n in range(1, Niter):
        theta[n] = theta[n-1] + gamma*np.sum(np.sum(x[n-1, :, :]*(y-theta[n-1]), axis = 1)*W)
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :, :] = x[n-1, ancestors, :]
        # MCMC move
        prop = rwm_proposal(x[n-1, :, :], W)
        x[n, :] = rwm_accept_multimodal_fast(x[n-1, :, :], prop, theta[n-2], gamma, y, n-1)
        # reweight  
        logW = (1-(1-gamma)**n)*ll_multimodal(theta[n-1], x[n, :, :], y) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_multimodal(theta[n-2], x[n, :, :], y) 
        W = rs.exp_and_normalise(logW)
    return theta, x, W


### Gaussian Mixture
# log-likelihood
def ll_gmm_alpha(theta, x, y, alpha):
    N = x.size
    ll = -(y-theta*x)**2/2-0.5*np.log(2*np.pi) +np.log((x==-1)+alpha*x)
    return ll  
# discrete proposal
def component_proposal_alpha(v, alpha):
    N = v.size
    arr_prop = 2*np.random.binomial(1, alpha, N)-1
#     arr_prop = 2*np.random.binomial(1, 0.5, N)-1
    return arr_prop
# accept/reject step
def accept_gmm_fast(v, prop, theta_current, gamma, data, n, alpha):
    log_acceptance = (1-(1-gamma)**n)*(ll_gmm_alpha(theta_current, prop, data, alpha) - ll_gmm_alpha(theta_current, v, data, alpha)) + 2*np.log(alpha/(1-alpha))*(prop-v)
    accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
    output = v.copy()
    output[accepted] = prop[accepted]
    return output
# SMC
def md_gmm_fast(y, gamma, Niter, N, th0, X0, alpha):
    ndata = y.size
    x = np.zeros((Niter, N))
    theta = np.zeros(Niter)
    theta[0] = th0
    x[0, :] = X0
    W = np.ones(N)/N
    for n in range(1, Niter):
        theta[n] = theta[n-1] + gamma*np.sum((y - theta[n-1]*x[n-1, :])*x[n-1, :]*W)
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :] = x[n-1, ancestors]
        # MCMC move
        prop = component_proposal_alpha(x[n-1,:], alpha)
        x[n, :] = accept_gmm_fast(x[n-1,:], prop, theta[n-2], gamma, y, n-1, alpha)
        # reweight  
        logW = (1-(1-gamma)**n)*ll_gmm_alpha(theta[n-1], x[n, :], y, alpha)+ np.log(0.5)*gamma*((1-gamma)**(n-1))
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_gmm_alpha(theta[n-2], x[n, :], y, alpha) 
        W = rs.exp_and_normalise(logW)
    return theta, x, W


### Stochastic Block Model
# log-likelihood
def ll_sbm(theta, x, y):
    N = x.size
    prior = np.zeros(N)
    ll = np.zeros((N))
    for i in range(N):
        if(x[i] == 1):
            prior[i] = np.log(theta[0])
            for j in range(N):
                if(j != i):
                    if(x[j] == 1):
                        ll[i] += y[i,j]*np.log(theta[1]) + (1-y[i,j])*np.log(1-theta[1])
                        ll[i] += y[j,i]*np.log(theta[1]) + (1-y[j,i])*np.log(1-theta[1])
                    if(x[j] == 2):
                        ll[i] += y[i,j]*np.log(theta[2]) + (1-y[i,j])*np.log(1-theta[2])
                        ll[i] += y[j,i]*np.log(theta[2]) + (1-y[j,i])*np.log(1-theta[2])
        if(x[i] == 2):
            prior[i] = np.log(1-theta[0])
            for j in range(N):
                if(j != i):
                    if(x[j] == 1):
                        ll[i] += y[i,j]*np.log(theta[2]) + (1-y[i,j])*np.log(1-theta[2])
                        ll[i] += y[j,i]*np.log(theta[2]) + (1-y[j,i])*np.log(1-theta[2])
                    if(x[j] == 2):
                        ll[i] += y[i,j]*np.log(theta[3]) + (1-y[i,j])*np.log(1-theta[3])
                        ll[i] += y[j,i]*np.log(theta[3]) + (1-y[j,i])*np.log(1-theta[3])
    return ll+prior
# discrete proposal
def component_proposal_sbm(v):
    N = v.size
    arr_prop = np.random.binomial(1, 0.5, N)+1
    return arr_prop
# accept/reject step
def accept_sbm_fast(v, prop, theta_current, gamma, data, n):
    log_acceptance = (1-(1-gamma)**n)*(ll_sbm(theta_current, prop, data) - ll_sbm(theta_current, v, data))
    accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
    output = v.copy()
    output[accepted] = prop[accepted]
    return output
def sbm_gradient_p(x, p):
    return (x == 1)/p - (x == 2)/(1-p)   
def sbm_gradient_nu(x, W, theta, y):
    N = x.size
    gradient = np.zeros(theta.size-1)
    for i in range(N):
        for j in range(N):
            if(j != i):
                if(x[i] == 1):
                    if(x[j] == 1):
                        gradient[0] += (y[i,j]/theta[1] - (1-y[i,j])/(1-theta[1]))*W[i]*W[j]
                        gradient[0] += (y[j,i]/theta[1] - (1-y[j,i])/(1-theta[1]))*W[i]*W[j]
                    if(x[j] == 2):
                        gradient[1] += (y[i,j]/theta[2] - (1-y[i,j])/(1-theta[2]))*W[i]*W[j]
                        gradient[1] += (y[j,i]/theta[2] - (1-y[j,i])/(1-theta[2]))*W[i]*W[j]
                if(x[i] == 2):
                    if(x[j] == 1):
                        gradient[1] += (y[i,j]/theta[2] - (1-y[i,j])/(1-theta[2]))*W[i]*W[j]
                        gradient[1] += (y[j,i]/theta[2] - (1-y[j,i])/(1-theta[2]))*W[i]*W[j]
                    if(x[j] == 2):
                        gradient[2] += (y[i,j]/theta[3] - (1-y[i,j])/(1-theta[3]))*W[i]*W[j]
                        gradient[2] += (y[j,i]/theta[3] - (1-y[j,i])/(1-theta[3]))*W[i]*W[j]
    return gradient
# SMC
def md_sbm_fast(y, gamma, Niter, N, th0, X0):
    x = np.zeros((Niter, N))
    theta = np.zeros((Niter, th0.size))
    theta[0, :] = th0
    x[0, :] = X0
    W = np.ones(N)/N
    kmeans = KMeans(n_clusters=2)
    for n in range(1, Niter):
        theta[n, 0] = theta[n-1,0]+gamma*np.sum(sbm_gradient_p(x[n-1,:].astype(int), theta[n-1, 0])*W)
        theta[n, 1:] = theta[n-1,1:]+gamma*sbm_gradient_nu(x[n-1,:].astype(int), W, theta[n-1, :], y)
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            x[n-1, :] = x[n-1, ancestors]
        # MCMC move
        kmeans.fit(y)
        prop = kmeans.labels_+1
        x[n, :] = accept_sbm_fast(x[n-1,:].astype(int), prop, theta[n-1, :], gamma, y, n-1)
        # reweight  
        logW = (1-(1-gamma)**n)*ll_sbm(theta[n-1, :], x[n, :].astype(int), y)+ np.log(0.5)*gamma*((1-gamma)**(n-1))
        if(n>1):
            logW = logW - (1-(1-gamma)**(n-1))*ll_sbm(theta[n-2, :], x[n, :].astype(int), y) 
        W = rs.exp_and_normalise(logW)
    return theta, x, W

# # ### Independent component analysis
# # log-likelihood
# def ll_ica(sigma, A, x, alpha, y):
#     d = y.shape[0]
#     N = y.shape[1]
#     loglikelihood = np.zeros(N)
#     for i in range(N):
#         mixture = alpha*logistic.pdf(x[i,:], scale = 1/2) + (1-alpha)*(x[i,:] == 0)
#         prior = np.sum(np.log(mixture))
#         loglikelihood[i] = -0.5*d*np.log(2*np.pi*sigma**2)-0.5*np.sum((y[:,i] - np.matmul(A, x[i,:]))**2, axis = 0)/sigma**2 + prior
#     return loglikelihood
# # accept/reject step
# def rwm_accept_ica_fast(v, prop, sigma_current, A_current, alpha, gamma, data, n):
#     log_acceptance = 0.5*(1-gamma)**n*np.sum(v**2 - prop**2, axis = 1)+(1-(1-gamma)**n)*(ll_ica(sigma_current, A_current, prop, alpha, data) - ll_ica(sigma_current, A_current, v, alpha, data))
#     accepted = np.log(np.random.uniform(size = v.shape[0])) <= log_acceptance
#     output = ssp.view_2d_array(v)
#     output[accepted, :] = prop[accepted, :]
#     return output

# # SMC
# def md_ica_fast(y, gamma, Niter, N, A0, sigma0, X0, alpha):
#     d = y.shape[0]
#     p = X0.shape[1]
#     x = np.zeros((Niter, N, p))
    
#     A = np.zeros((Niter, d, p))
#     sigma = np.zeros((Niter))

#     A[0,:,:] = A0
#     sigma[0] = sigma0

#     x[0, :, :] = X0
#     W = np.ones(N)/N
#     for n in range(1, Niter):
#         sigma[n] = sigma[n-1] + gamma*np.sum(ica_gradient_sigma(A[n-1, :, :], sigma[n-1], y, x[n-1, :, :])*W)
#         A[n, :, :] = A[n-1, :, :] + gamma*np.sum(ica_gradient_A(A[n-1, :, :], sigma[n-1], y, x[n-1, :, :]).T*W, axis = 2).T
#         if (n > 1):
#             # resample
#             ancestors = rs.resampling('stratified', W)
#             x[n-1, :, :] = x[n-1, ancestors, :]
#         # MCMC move
#         prop = rwm_proposal(x[n-1, :, :], W)
#         x[n, :, :] = rwm_accept_ica_fast(x[n-1, :, :], prop, sigma[n-2], A[n-2, :, :], alpha, gamma, y, n-1)
#         # reweight
#         logW = (1-(1-gamma)**n)*ll_ica(sigma[n-1], A[n-1, :, :], x[n, :, :], alpha, y) + 0.5*gamma*(1-gamma)**(n-1)*np.sum(x[n, :, :]**2, axis = 1)
#         if(n>1):
#             logW = logW - (1-(1-gamma)**(n-1))*ll_ica(sigma[n-2], A[n-2, :, :], x[n, :, :], alpha, y) 
#         W = rs.exp_and_normalise(logW)
#     return A, sigma, x, W
