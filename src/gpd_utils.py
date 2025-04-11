import numpy as np

def quantile_GPD(F, u, sigma, gamma):
    """
    Calculate the quantile of GPD for a certain probability

    Args:
        F (_type_): _description_
        u (_type_): _description_
        sigma (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Gumbel case
    if np.abs(gamma) < 1e-8:
        q = u - sigma*np.log(1-np.log(F))

    # General case
    else:
        q = u - (1-(1-F)**(-gamma))*sigma/gamma

    return q

def dq_gpd(F, u, sigma, gamma):
    """
    Calculates the dertivative of quantiles of POT
    """
    Dq = np.zeros((3,len(F)))
    # Gumbel case
    if np.abs(gamma) < 1e-8:
        Dqu = np.zeros(len(F))
        Dqsigma = -np.log(1-np.log(F))
        Dqxi = np.zeros(len(F))
    # General case
    else:
        Dqu = np.zeros(len(F))
        Dqsigma = -(1-(1-F)**(-gamma))/gamma
        Dqxi = -(sigma/((gamma**2)*(1-F)**(gamma)))*(gamma*np.log(1-F)-(1-F)**(gamma)+1)
        # Dqxi = sigma*(1-(1-F)**(-gamma)*(1+gamma*np.log(1-F)))/(gamma**2)

    Dq[0,:] = Dqu
    Dq[1,:] = Dqsigma*sigma
    Dq[2,:] = Dqxi

    return Dq

def nll_gpd(data, p):
    """
    Negative Loglikelihood of Stationary GPD distribution.
    Usefull to compute the variance-covariance matrix using the hessian from numdifftools

    Args:
        data (_type_): data
        p (list): parameters of GPD distribution [threshold, scale, shape] 

    Returns:
        loglikelihood (np.array): negative loglikelihood value
    """

    u = p[0]       # Location
    sigma = p[1]   # Scale
    xi = p[2]      # Shape

    # N = len(data)
    exceedances = data[data > u] 
    N = len(exceedances)
    
    # Gumbel 
    if np.abs(xi) < 1e-8:
        expr = (data-u)/sigma
        expr = np.maximum(expr, 1e-5)
        return (N*np.log(sigma) + np.sum(expr))

    # Weibull-Frechet 
    else:
        expr = 1+xi*((data-u)/sigma)
        # expr = np.maximum(expr, 1e-5)
        return (N*np.log(sigma) + (1+1/xi)*np.sum(np.log(expr)))
    

def nll_gpd_hessian(data, p):
    """
    Negative Loglikelihood of Stationary GPD distribution.
    Usefull to compute the variance-covariance matrix using the hessian from numdifftools

    Args:
        data (_type_): data
        p (list): parameters of GPD distribution [threshold, scale, shape] 

    Returns:
        loglikelihood (np.array): negative loglikelihood value
    """

    u = p[0]       # Location
    sigma = p[1]   # Scale
    xi = p[2]      # Shape

    N = len(data)
    
    # Gumbel 
    if np.abs(xi) < 1e-8:
        expr = (data-u)/sigma
        expr = np.maximum(expr, 1e-5)
        f = (N*np.log(sigma) + np.sum(expr))

    # Weibull-Frechet 
    else:
        expr = 1+xi*((data-u)/sigma)
        expr = np.maximum(expr, 1e-5)
        f = (N*np.log(sigma) + (1+1/xi)*np.sum(np.log(expr)))


    # Jacobian
    j = np.zeros(3)
    z = (data-u)/sigma
    expr = 1+xi*z
    j[0] = -(1+1/xi)*np.sum(xi/(sigma*expr))
    j[1] = N/sigma - (1+1/xi)*np.sum(xi*z/(sigma*expr))
    j[2] = -1/(xi**2)*np.sum(np.log(expr)) + (1+1/xi)*np.sum(z/expr)

    # Hessian
    hess = np.zeros((3,3))
    # Diagonal elements
    hess[0,0] = (1+1/xi)*np.sum(xi**2/((sigma**2)*(expr**2)))
    hess[1,1] = -N/sigma**2 + (1+1/xi)*np.sum(2*xi*z/(sigma**2*expr)- xi**2*z**2/(sigma**2*expr**2))
    hess[2,2] = 2/(xi**3)*np.sum(np.log(expr))-2/(xi**2)*np.sum(z/expr)-(1+1/xi)*np.sum(z**2/(expr**2))

    # Cross 
    hess[0,1] = (1+1/xi)*np.sum(xi**2*z/(sigma**2*expr**2))
    hess[1,0] = (1+1/xi)*np.sum(xi**2*z/(sigma**2*expr**2))
    
    hess[0,2] = 1/(xi**2)*np.sum(1/expr)-(1+1/xi)*np.sum(z/(sigma*expr**2))
    hess[2,0] = 1/(xi**2)*np.sum(1/expr)-(1+1/xi)*np.sum(z/(sigma*expr**2))

    hess[1,2] = (1+1/xi)*np.sum(z/expr**2*(xi*z/sigma)) - 1/(xi**2)*np.sum(xi*z/(sigma*expr))
    hess[2,1] = (1+1/xi)*np.sum(z/expr**2*(xi*z/sigma)) - 1/(xi**2)*np.sum(xi*z/(sigma*expr))
    
    return f, j, hess