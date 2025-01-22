import numpy as np
from scipy.stats.contingency import crosstab

### Stochastic Block Model -- Poisson
def ll_poisson_sbm(theta, x, y, i):
    N = x.size
    ll = 0
    for j in range(N):
        if(x[i] == 0):
            prior = np.log(theta[0])
            if(j != i):
                if(x[j] == 0):
                    ll += y[i,j]*np.log(theta[1]) - theta[1] 
                    ll += y[j,i]*np.log(theta[1]) - theta[1] 
                if(x[j] == 1):
                    ll += y[i,j]*np.log(theta[3]) - theta[3] 
                    ll += y[j,i]*np.log(theta[2]) - theta[2] 
        if(x[i] == 1):
            prior = np.log(1-theta[0])
            if(j != i):
                if(x[j] == 0):
                    ll += y[i,j]*np.log(theta[2]) - theta[2] 
                    ll += y[j,i]*np.log(theta[3]) - theta[3] 
                if(x[j] == 1):
                    ll += y[i,j]*np.log(theta[4]) - theta[4]  
                    ll += y[j,i]*np.log(theta[4]) - theta[4] 
    return ll+prior
def poisson_sbm_saem_proposal(data, v, theta_current):
    N = v.size
    output = v.copy()
    for i in range(N):
        prop = np.random.binomial(1, 0.5, 1)
        if(prop != v[i]):
            prop_full = v.copy()
            prop_full[i] = prop
            log_acceptance = ll_poisson_sbm(theta_current, prop_full, data, i) - ll_poisson_sbm(theta_current, v, data, i)
            if (np.log(np.random.uniform(size = 1)) <= log_acceptance):
                output[i] = prop
    return output
def poisson_sbm_saem_sufficient_stat(x, y):
    N = x.size
    s1 = np.sum(x == 0)
    s2 = np.zeros(4)
    s3 = np.zeros(4)
    for i in range(N):
        for j in range(N):
            if(j != i):
                if(x[i] == 0):
                    if(x[j] == 0):
                        s2[0] += y[i,j]
                        s3[0] += 1
                    if(x[j] == 1):
                        s2[2] += y[i,j]
                        s3[2] += 1
                if(x[i] == 1):
                    if(x[j] == 0):
                        s2[1] += y[i,j]
                        s3[1] += 1
                    if(x[j] == 1):
                        s2[3] += y[i,j]
                        s3[3] += 1
    return s1, s2, s3
def poisson_sbm_saem_mle(s1, s2, s3, N):
    pq = s1/N
    nu = s2/s3
    return np.append(pq, nu)

### Stochastic Block Model -- Bernoulli
def ll_sbm(theta, x, y, i):
    N = x.size
    ll = 0
    for j in range(N):
        if(x[i] == 0):
            prior = np.log(theta[0])
            if(j != i):
                if(x[j] == 0):
                    ll += y[i,j]*np.log(theta[1]) + (1-y[i,j])*np.log(1-theta[1])
                    ll += y[j,i]*np.log(theta[1]) + (1-y[j,i])*np.log(1-theta[1])
                if(x[j] == 1):
                    ll += y[i,j]*np.log(theta[3]) + (1-y[i,j])*np.log(1-theta[3])
                    ll += y[j,i]*np.log(theta[2]) + (1-y[j,i])*np.log(1-theta[2])
        if(x[i] == 1):
            prior = np.log(1-theta[0])
            if(j != i):
                if(x[j] == 0):
                    ll += y[i,j]*np.log(theta[2]) + (1-y[i,j])*np.log(1-theta[2])
                    ll += y[j,i]*np.log(theta[3]) + (1-y[j,i])*np.log(1-theta[3])
                if(x[j] == 1):
                    ll += y[i,j]*np.log(theta[4]) + (1-y[i,j])*np.log(1-theta[4])
                    ll += y[j,i]*np.log(theta[4]) + (1-y[j,i])*np.log(1-theta[4])
    return ll+prior
def sbm_saem_proposal(data, v, theta_current):
    N = v.size
    output = v.copy()
    for i in range(N):
        prop = np.random.binomial(1, 0.5, 1)
        if(prop != v[i]):
            prop_full = v.copy()
            prop_full[i] = prop
            log_acceptance = ll_sbm(theta_current, prop_full, data, i) - ll_sbm(theta_current, v, data, i)
            if (np.log(np.random.uniform(size = 1)) <= log_acceptance):
                output[i] = prop
    return output
def sbm_saem_sufficient_stat(x, y):
    N = x.size
    s1 = np.sum(x == 0)
    s2 = np.zeros(4)
    s3 = np.zeros(4)
    for i in range(N):
        for j in range(N):
            if(j != i):
                if(x[i] == 0):
                    if(x[j] == 0):
                        s2[0] += y[i,j]
                        s3[0] += (1-y[i,j])
                    if(x[j] == 1):
                        s2[2] += y[i,j]
                        s3[2] += (1-y[i,j])
                if(x[i] == 1):
                    if(x[j] == 0):
                        s2[1] += y[i,j]
                        s3[1] += (1-y[i,j])
                    if(x[j] == 1):
                        s2[3] += y[i,j]
                        s3[3] += (1-y[i,j])
    return s1, s2, s3
def sbm_saem_mle(s1, s2, s3, N):
    pq = s1/N
    nu = s2/(s2+s3)
    return np.append(pq, nu)

def sbm_ari(ztrue, zest):
    nodes = ztrue.size
    ctab = crosstab(zest, ztrue)[1]
    rowsum = np.sum(ctab, axis = 1)
    colsum = np.sum(ctab, axis = 0)
    num = np.sum(ctab*(ctab-1)/2) - np.sum(rowsum*(rowsum-1)/2)*np.sum(colsum*(colsum-1)/2)/(nodes*(nodes-1)/2)
    den = 0.5*(np.sum(rowsum*(rowsum-1)/2)+np.sum(colsum*(colsum-1)/2))- np.sum(rowsum*(rowsum-1)/2)*np.sum(colsum*(colsum-1)/2)/(nodes*(nodes-1)/2)
    return num/den

def se_sbm(theta, pi, probs):
    se = np.zeros(5)
    rse = np.zeros(5)
    true_zero = np.argmin((theta[0] - pi)**2)
    switched = 0
    if (true_zero == 0):
        se[0] = (theta[0] - pi[0])**2
        rse[0] = se[0]/pi[0]**2
        se[1:] = (theta[1:]-probs.flatten())**2
        rse[1:] = se[1:]/probs.flatten()**2
    else:
        se[0] = (theta[0] - pi[1])**2
        rse[0] = se[0]/pi[1]**2
        tmp = theta[1:]
        se[1:] = (tmp[::-1]-probs.flatten())**2
        rse[1:] = se[1:]/probs.flatten()**2
        switched = 1
    return se, rse, switched