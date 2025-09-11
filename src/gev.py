import numpy as np
from scipy.optimize import minimize


class gev:
    @staticmethod
    def supp(loc = 0, scale = 1, shape = 0):

        if np.min(scale) < 0:
            ValueError("Invalid scale")

        if shape == 0:
            return (-np.inf, np.inf)
        if shape > 0:
            return (loc - scale/shape, np.inf)
        else:
            return (-np.inf, loc - scale/shape)

    @staticmethod
    def pdf(x, loc = 0, scale = 1, shape = 0, log=False):
        
        if np.min(scale) < 0:
            ValueError("Invalid scale")

        x = (x - loc)/scale

        if shape == 0:
            d = np.log(1/scale) - x - np.exp(-x)

        else:
            xx = 1 + shape*x
            d = np.log(1/scale) - (xx**(-1/shape)) - (1/shape - 1) * np.log(xx)
        
        return np.exp(d) 
    
    @staticmethod
    def logpdf(x, loc = 0, scale = 1, shape = 0):
        
        if np.min(scale) < 0:
            ValueError("Invalid scale")

        x = (x - loc)/scale

        if shape == 0:
            d = np.log(1/scale) - x - np.exp(-x)

        else:
            xx = 1 + shape*x
            d = np.log(1/scale) - (xx**(-1/shape)) - (1/shape - 1) * np.log(xx)
        
        return d 

    @staticmethod
    def cdf(q, loc=0, scale=1, shape=0):

        if np.min(scale) < 0:
            ValueError("Invalid scale")

        q = (q - loc) / scale

        if shape == 0:
            return np.exp(-np.exp(-q))
        
        else:
            return np.exp(np.maximum(1+shape*q, 0)**(-1/shape))

    @staticmethod
    def quanf(p, loc=0, scale=1, shape=0):
        
        if (np.min(p) <= 0 or np.max(p) >= 1):
            ValueError("p must contain probabilities between (0,1)")

        if np.min(scale) < 0:
            ValueError("Invalid scale")

        if shape == 0:
            return loc - scale * np.log(-np.log(p))
        else:
            return loc + scale *((-np.log(p))**(-shape)-1)/shape
    
    @staticmethod
    def loglike(x, loc, scale, shape):

        if np.min(scale) < 0:
            ValueError("Invalid scale")
        
        x = np.atleast_1d(x)
        n = x.size

        x = (x - loc)/scale

        if shape == 0:
            return -n*np.log(scale) - np.sum(x) - np.sum(np.exp(-x))
        else:
            xx = 1+shape*x
            return -n*np.log(scale) - (1/shape + 1)*np.sum(np.log(xx)) - np.sum(xx**(-1/shape)) 
        

    @staticmethod
    def random(size=1, loc=0, scale=1, shape=0):

        u = np.random.uniform(0,1,size=size)

        return gev.quanf(u, loc, scale, shape)
    
    @staticmethod
    def fit(x, method="mle"):
        
        if method.lower() != "mle":
            raise ValueError(f"{method} method not supported yet")
        
        else: 
            # Function loglikelihood
            neg_log_like = lambda p: - gev.loglike(x, p[0], p[1], p[2])

            bnds = ((None, None), (1e-8, None), (None, None))
            res = minimize(
                fun = neg_log_like,
                x0 = np.array([np.mean(x), np.std(x), 0.1]),
                method = "SLSQP",
                bounds=bnds
            )

            if not res.success:
                Warning("The minimization do not converge")
                return res
            
            return {
                'x': res.x,
                'fun': res.fun,
                'success': res.success
            }




        
