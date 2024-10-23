import numpy as np
from scipy.stats import multivariate_normal, logistic, geom

### Gradients and marginal likelihood of ICA model
def logICA_gradient_sigma(A, sigma, Y, z):
    """ Gradient of likelihood w.r.t. sigma for LogICA model
    A: matrix of components (parameter)
    sigma: standard deviation of likelihood (parameter)
    Y: current data point
    z: current latent state
    mu: (optional) mean parameter
    """
    d = Y.size
    log_gradient = np.sum((Y - np.matmul(A, z))**2)/sigma**3 - d/sigma
    return log_gradient

def logICA_gradient_A(A, sigma, Y, z):
    """ Gradient of likelihood w.r.t. A for LogICA model
    A: matrix of components (parameter)
    sigma: standard deviation of likelihood (parameter)
    mu: mean of data (parameter)
    Y: current data point
    z: current latent state
    """
    tmp = np.outer(Y - np.matmul(A, z), z)
    gradient = np.sign(tmp) * np.exp(np.log(np.abs(tmp)) - np.log(sigma**2))
    return gradient
    
def censored_logICA_marginal_likelihood(A, sigma, alpha, data, H = 100):
    """ Marginal likelihood approximation for censored LogICA model
    A: matrix of components (parameter)
    sigma: standard deviation of likelihood (parameter)
    data: current data point
    H: number of samples to approximate marginal likelihood
    """
    p = A.shape[1]
    d = A.shape[0]
    z_covariance = np.linalg.inv(np.matmul(np.transpose(A), A))
    z_mean = np.matmul(np.matmul(z_covariance, np.transpose(A)), data)
    constant = (2*np.pi*sigma**2)**((p-d)/2)*np.linalg.det(np.matmul(np.transpose(A), A))**(-1/2)
    weight = constant * np.exp(-0.5*(np.linalg.norm(data, 2)**2-np.matmul(data, np.matmul(A, z_mean.T)))/sigma**2)
    z = np.random.multivariate_normal(mean = z_mean, cov = sigma**2*z_covariance, size = H)
    mixture = alpha*logistic.pdf(z, scale = 1/2) + (1-alpha)*(z == 0)
    marginal_likelihood = np.prod(mixture, axis = 1)*weight
    return  np.mean(marginal_likelihood)
### SAEM
    
def censored_logICA_sample_posterior(z_past, A, sigma, alpha, data):
    """ Metropolis within Gibbs to sample from posterior of censored LogICA model
    z_past: current value of the chain
    A: matrix of components (parameter)
    sigma: standard deviation of likelihood (parameter)
    alpha: censoring parameter
    data: current data point
    """
    p = A.shape[1]
    d = A.shape[0]
    component = np.random.randint(p, size=1)
    proposal = np.copy(z_past)
    proposal[component] = np.random.binomial(1, alpha, size = 1)*np.random.logistic(scale = 1/2, size = 1)
    likelihood_past = np.sum((data - np.matmul(A, z_past))**2)
    likelihood_proposal = np.sum((data - np.matmul(A, proposal))**2)
    log_acceptance_ratio = (likelihood_past - likelihood_proposal)/(2*sigma**2)
    u = np.random.uniform(size = 1)
    if (u <= np.exp(log_acceptance_ratio)):
        out = proposal
    else:
        out = np.copy(z_past)
    return out
    
def censored_logICA_sample_posterior_ar(z_past, A, sigma, alpha, data):
    """ Metropolis within Gibbs to sample from posterior of censored LogICA model
    z_past: current value of the chain
    A: matrix of components (parameter)
    sigma: standard deviation of likelihood (parameter)
    alpha: censoring parameter
    data: current data point
    """
    p = A.shape[1]
    d = A.shape[0]
    component = np.random.randint(p, size=1)
    proposal = np.copy(z_past)
    proposal[component] = np.random.binomial(1, alpha, size = 1)*np.random.logistic(scale = 1/2, size = 1)
    likelihood_past = np.sum((data - np.matmul(A, z_past))**2)
    likelihood_proposal = np.sum((data - np.matmul(A, proposal))**2)
    log_acceptance_ratio = (likelihood_past - likelihood_proposal)/(2*sigma**2)
    u = np.random.uniform(size = 1)
    if (u <= np.exp(log_acceptance_ratio)):
        accepted = 1
        out = proposal
    else:
        accepted = 0
        out = np.copy(z_past)
    return out, accepted

def logICA_sufficient_statistic(z, data):
    """ Compute sufficient statistic for LogICA model
    z: sample from posterior
    data: current data point
    """
    z_tmp = np.atleast_2d(z)
    data_tmp = np.atleast_2d(data)
    s1 = np.matmul(z_tmp.T, z_tmp)
    s2 = np.matmul(data_tmp.T, z_tmp)
    return s1, s2

def logICA_mle(s1, s2, Y):
    """ Update parameters for LogICA model (M step)
    s1: first sufficient statistic
    s2: second sufficient statistic
    Y: data set
    """
    d = Y.shape[0]
    A = np.matmul(s2, np.linalg.inv(s1))
    sigma = np.sqrt((np.mean(np.sum(Y**2, axis = 0))-2*np.sum(A*s2) + np.sum(np.matmul(A.T, A) * s1))/d)
    return A, sigma
