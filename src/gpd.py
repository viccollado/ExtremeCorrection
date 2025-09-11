import numpy as np


class gpd:
    @staticmethod
    def pdf(x, loc = 0, scale = 1, shape = 0):
        ...

    @staticmethod
    def cdf(q, loc=0, scale=1, shape=0):
        if np.min(scale) < 0:
            ValueError("Invalid scale")

        q = np.maximum(q - loc, 0) / scale

        if shape == 0:
            return 1 - np.exp(-q)
        else:
            return 1- np.maximum(1+shape*q, 0)**(-1/shape)
        
    @staticmethod
    def ppf(p, loc=0, scale=1, shape=0):
        
        if (np.min(p) <= 0 or np.max(p) >= 1):
            ValueError("p must contain probabilities between (0,1)")

        if np.min(scale) < 0:
            ValueError("Invalid scale")

        if shape == 0:
            return loc - scale * np.log(p)
        else:
            return loc + scale *(p**(-shape)-1)/shape
    
    @staticmethod
    def loglike():
        ...

