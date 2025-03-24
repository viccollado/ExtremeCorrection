import numpy as np

def q_pot(F, u, lam, sigma, gamma):
    """
    Calculate the quantile of POT for a certain probability

    Args:
        F (_type_): _description_
        u (_type_): _description_
        lamb (_type_): _description_
        sigma (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    if np.abs(gamma) < 1e-8:
        q = u - sigma*np.log(-np.log(F)(lam))

    else:
        q = u - (1-(-np.log(F)/lam)**(-gamma))*sigma/gamma

    return q

def dq_pot(F, u, lam, sigma, gamma):
    """
    Calculates the dertivative of quantiles of POT
    """
    Dq = np.zeros((4,len(F)))
    # Gumbel case
    if np.abs(gamma) < 1e-8:
        Dqu = np.zeros(len(F))
        Dqlam = sigma/lam
        Dqsigma = -np.log(-np.log(F)/lam)
        Dqxi = np.zeros(len(F))
    # General case
    else:
        Dqu = np.zeros(len(F))
        Dqlam = sigma/lam*(-np.log(F)/lam)**(-gamma)
        Dqsigma = -(1-(-np.log(F)/lam)**(-gamma))/gamma
        Dqxi = sigma*(1-(-np.log(F)/lam)**(-gamma)*(1+gamma*np.log(-np.log(F)/lam)))/(gamma*gamma)


    Dq[0,:] = Dqu
    Dq[1,:] = Dqlam
    Dq[2,:] = Dqsigma*sigma
    Dq[3,:] = Dqxi

    return Dq

def nll_pot(data, p):
    """
    Negative Loglikelihood of Stationary POT distribution.
    Usefull to compute the variance-covariance matrix using the hessian from numdifftools

    Args:
        data (_type_): data
        p (list): parameters of POT distribution [threshold, scale, shape] 

    Returns:
        loglikelihood (np.array): negative loglikelihood value
    """


    u = p[0]       # Location
    lam = p[1]     # Poisson 
    sigma = p[2]   # Scale
    xi = p[3]      # Shape

    n = len(data)
    exceedances = data[data > u] 
    N = len(exceedances)

    
    # Gumbel 
    if np.abs(xi) < 1e-8:
        expr = (exceedances-u)/sigma
        expr = np.maximum(expr, 1e-5)
        return (- N*np.log(lam) + n*lam + N*np.log(sigma) + np.sum(expr))

    # Weibull-Frechet 
    else:
        expr = 1+xi*((exceedances-u)/sigma)
        expr = np.maximum(expr, 1e-5)
        return (- N*np.log(lam) + n*lam + N*np.log(sigma) + (1+1/xi)*np.sum(np.log(expr)))
    

def cdf_pot(data, u, lam, sigma, gamma):
    """
    CDF of POT

    Args:
        data (_type_): _description_
        u (_type_): _description_
        lam (_type_): _description_
        sigma (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    if np.abs(gamma) < 1e-8:
        expr = (data-u)/sigma
        return np.exp(-lam*np.exp(-expr))
    
    else:
        expr = 1+gamma*((data-u)/sigma)
        return np.exp(-lam*expr**(-1/gamma))