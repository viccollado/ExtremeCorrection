import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA
from sklearn.mixture import GaussianMixture
from scipy.optimize import root_scalar
from tqdm import tqdm


def gmm_cdf(gmm, x):
    """
    Compute the CDF of a fitted 1D Gaussian Mixture Model.
    
    Args:
        gmm: A GaussianMixture object with attributes `means_`, `covariances_`, and `weights_`.
        x (float or array-like): Quantile(s) where the CDF is evaluated.
        
    Returns:
        float or np.array: The CDF value(s) corresponding to x.
    """
    x = np.atleast_1d(x)  # Ensure x is an array even if a scalar is provided
    means = gmm.means_.flatten()  # Shape: (n_components,)
    stds = np.sqrt(gmm.covariances_.flatten())  # Convert variances to std deviations
    
    # Compute the CDF for each component at all quantiles (broadcasting: shape (len(x), n_components))
    component_cdfs = stats.norm.cdf(x[:, None], loc=means, scale=stds)
    result = np.sum(component_cdfs * gmm.weights_, axis=1)
    
    # If the input was a scalar, return a scalar instead of an array.
    if result.size == 1:
        return result[0]
    return result

def gmm_quantile(gmm, probabilities):
    """
    Compute quantiles for a fitted 1D Gaussian Mixture Model given target probabilities.
    
    Args:
        gmm: A GaussianMixture object with attributes `means_`, `covariances_`, and `weights_`.
        probabilities (float or array-like): The probability (or probabilities) for which to compute quantiles.
        
    Returns:
        float or np.array: The quantile(s) corresponding to the provided probability(ies).
    """
    probabilities = np.atleast_1d(probabilities)
    quantiles = np.zeros_like(probabilities, dtype=float)
    
    for i, p in enumerate(tqdm(probabilities)):
        # Use a root-finding algorithm (bisect method) to invert the CDF.
        result = root_scalar(
            lambda x: gmm_cdf(gmm, x) - p,
            # bracket=[-1, 20],  # Adjust bracket if necessary for your data
            method='newton',
            x0=0.5
        )
        quantiles[i] = result.root

    # Return a scalar if the input was scalar.
    if quantiles.size == 1:
        return quantiles[0]
    return quantiles


class ArmaSimulation():

    def __init__(
        self,
        data: pd.DataFrame, 
        var: str, 
        freq: float = 365.25, 
        arma_order:tuple = (1,0,1),
        distribution:str="ecdf", 
        n_components: int = 4
    ):
        
        self.pit_data = data[var].values
        self.sort_idx = np.argsort(self.pit_data)
        self.n_pit = self.pit_data.size
        self.var = var
        self.freq = freq
        self.order = arma_order
        self.distribution = distribution.lower()
        self.n_components = n_components    # For the gaussian mixture distribution
        self.ar, self.ma, self.std_res = self.ARMAadjust()     # Fit ARIMA model

    @property
    def _ecdf_pit(self):
        """
        Empirical Distribution Function de point-in-time data

        Returns:
            _type_: _description_
        """
        return np.arange(1, self.n_pit+1)/(self.n_pit+1)
    
    def _pit_inverse(self, probs):
        """
        Compute the inverse of fitted Point-in-time distribution (quantile function) for given probabilities.

        Args:
            probs (_type_): _description_

        Returns:
            quantiles: _description_
        """
        if self.distribution == "norm":
            return stats.norm.ppf(probs, loc=self.params[0], scale=self.params[1])
        
        elif self.distribution == "lognorm":
            return stats.lognorm.ppf(probs, s=self.params[0], loc=self.params[1], scale=self.params[2])
        
        elif self.distribution == "gaussmix":
            return gmm_quantile(self.gmm, probs)
        
        elif self.distribution == "ecdf":
            return np.interp(probs, self._ecdf_pit, self.pit_data[self.sort_idx])
    
    def _pit_fit(self):
        """
        Fit a Normal or Lognormal distribution to the Point-in-time data

        Returns:
            _type_: _description_
        """

        if self.distribution == "norm":
            params = stats.norm.fit(self.pit_data)
        
        elif self.distribution == "lognorm":
            params = stats.lognorm.fit(self.pit_data)
        
        elif self.distribution == "gaussmix":
            self.gmm = GaussianMixture(n_components=4, covariance_type='diag', tol=1e-6, reg_covar=1e-6, max_iter=1000, random_state=42)
            self.gmm.fit(self.pit_data.reshape(-1, 1))
            params = None

        elif self.distribution == "ecdf":
            params = None

        else: 
            params = None
            raise ValueError("Insert a proper distribution ('norm', 'lognorm', 'gaussmix or 'ecdf').")

        self.params = params
    

    def _qnorm_pit(self):
        """
        Serie temporal transformada usando la transformación de rosenblatt para las probabilidades empíricas
        """
        
        if self.distribution == "norm":
            z_hist = stats.norm.ppf(stats.norm.cdf(self.pit_data, loc=self.params[0], scale=self.params[1]), loc=0, scale=1)

        elif self.distribution == "lognorm": 
            z_hist = stats.norm.ppf(stats.lognorm.cdf(self.pit_data, s=self.params[0], loc=self.params[1], scale=self.params[2]), loc=0, scale=1)

        elif self.distribution == "gaussmix":
            z_hist = stats.norm.ppf(gmm_cdf(self.gmm, self.pit_data), loc=0, scale=1)

        elif self.distribution == "ecdf":
            z_hist = stats.norm.ppf(self._ecdf_pit[self.sort_idx], loc=0, scale=1)

        self.qnorm_pit = z_hist 
    
    def ARMAadjust(self):
        """
        Ajuste ARMA a la serie transformada para obtener los parameros AR, MA y la desviación estándar de los residuos (sigma_eps)
        """
        self._pit_fit()     # Fit the point-in-time distribution
        self._qnorm_pit()   # Compute the temporal series of point-in-time (rosenblatt transformation)

        model = ARIMA(self.qnorm_pit, order=self.order, trend="n")  # ARMA(1,1) is ARIMA(1,0,1)
        result = model.fit()

        ar_param = result.arparams              # AR parameter
        ma_param = result.maparams              # MA parameter
        std_res = np.std(result.resid)          # Standard Deviation of residuals
        self.arma_result = result
        return ar_param, ma_param, std_res

    def generate_arma_process(self, noise, yp=None, epsiq=None):
        """
        Generate an ARMA(p, q) time series using provided AR, MA coefficients, and noise.
        
        Parameters:
        - ar_params : list -> AR coefficients (exclude the leading 1)
        - ma_params : list -> MA coefficients (exclude the leading 1)
        - noise : np.array -> Pre-generated normal errors
        - constant : float -> Constant term for the process
        
        Returns:
        - y : np.array -> Simulated ARMA series
        """
        n = len(noise)
        p, q = self.order[0], self.order[2]

        if yp is None:
            yp = np.zeros(p)
        if epsiq is None:
            epsiq = np.zeros(q)
        
        # Initialize series with zeros
        yt = np.zeros(n)
        
        # Generate ARMA process
        for t in range(n):
            yaux = noise[t]

            # MA component
            if q > 0:
                yaux += np.dot(self.ma[:q], epsiq[:q])
            epsiq = np.roll(epsiq, 1)
            epsiq[0] = noise[t]

            si = yaux

            # AR Component
            if p > 0:
                yaux += np.dot(self.ar[:p], yp[:p])
            yp = np.roll(yp, 1)
            yp[0] = yaux

            yt[t] = yaux
            
        return yt

    def generate_sim(self, ny_sim: int, yp=None, epsiq=None):
        """
        Generar la simulación en base al ARMA de longitud nsim, generando primero residuos con media 0 y desviación estandar la obtenida en ARMAadjust (sigma_eps)

        Args:
            ny_sim (int): Number of years of simulation
        """
        
        # Generate random normal errors with mean 0 and std = std_res 
        eps_sim = np.random.normal(loc=0, scale=self.std_res, size=int(ny_sim*self.freq))
        self.eps_sim = eps_sim
        # Introducir los errores en el ARMA y obtener z_sim
        z_sim = self.generate_arma_process(eps_sim, yp, epsiq)
        self.z_sim = z_sim
        # Calcular la serie temporal de probabilidades u_sim de z_sim usando la normal (u_sim = Phi(z_sim))
        u_sim = stats.norm.cdf(z_sim, loc=0, scale=1)   # Uniformly distributed
        self.u_sim = u_sim
        # Usar la inversa de la función empírica de point-in-time para calcular la serie simulada nueva
        x_sim = self._pit_inverse(u_sim)

        # Devolver la serie simulada en la que hay que aplicar la corrección
        return x_sim
    
    