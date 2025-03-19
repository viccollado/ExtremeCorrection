import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA

class ArmaSimulation():

    def __init__(self, data: pd.DataFrame, var: str, freq: float = 365.25):
        
        self.pit_data = data[var].values
        self.sort_idx = np.argsort(self.pit_data)
        self.n_pit = self.pit_data.size
        self.var = var
        self.freq = freq
        self.ar, self.ma, self.std_res = self.ARMAadjust()     # Fit ARIMA model

    @property
    def _ecdf_pit(self):
        """
        Empirical Distribution Function de point-in-time data

        Returns:
            _type_: _description_
        """
        return np.arange(1, self.n_pit+1)/(self.n_pit+1)
    
    def _ecdf_inverse(self, probs):
        """
        Compute the inverse ECDF (quantile function) for given probabilities.

        Args:
            probs (_type_): _description_

        Returns:
            quantiles: _description_
        """
        return np.interp(probs, self._ecdf_pit, self.pit_data[self.sort_idx])
    
    @property
    def _qnorm_pit(self):
        """
        Serie temporal transformada usando la transformación de rosenblatt para las probabilidades empíricas
        """
        return stats.norm.ppf(self._ecdf_pit[self.sort_idx], loc=0, scale=1)
    
    def ARMAadjust(self):
        """
        Ajuste ARMA a la serie transformada para obtener los parameros AR, MA y la desviación estándar de los residuos (sigma_eps)

        """
        model = ARIMA(self._qnorm_pit, order=(1, 0, 1), trend="n")  # ARMA(1,1) is ARIMA(1,0,1)
        result = model.fit()

        ar_param = result.params[0]             # AR parameter
        ma_param = result.params[1]             # MA parameter
        std_res = np.sqrt(result.params[2])   # Standard Deviation of residuals
        return ar_param, ma_param, std_res

    def generate_arma_process(self, noise):
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
        
        # Initialize series with zeros
        y = np.zeros(n)
        
        # Generate ARMA process
        for t in range(n):
            # AR component
            ar_component = self.ar * y[t - 1]
            # MA component
            ma_component = self.ma * noise[t - 1]
            
            # ARMA process equation
            y[t] = ar_component + noise[t] + ma_component
            
        return y

    def generate_sim(self, ny_sim: int):
        """
        Generar la simulación en base al ARMA de longitud nsim, generando primero residuos con media 0 y desviación estandar la obtenida en ARMAadjust (sigma_eps)

        Args:
            ny_sim (int): Number of years of simulation
        """
        
        # Generate random normal errors with mean 0 and std = std_res 
        eps_sim = np.random.normal(loc=0, scale=self.std_res, size=int(ny_sim*self.freq))

        # Introducir los errores en el ARMA y obtener z_sim
        z_sim = self.generate_arma_process(eps_sim)

        # Calcular la serie temporal de probabilidades u_sim de z_sim usando la normal (u_sim = Phi(z_sim))
        u_sim = stats.norm.cdf(z_sim, loc=0, scale=1)

        # Usar la inversa de la función empírica de point-in-time para calcular la serie simulada nueva
        x_sim = self._ecdf_inverse(u_sim)

        # Devolver la serie simulada en la que hay que aplicar la corrección
        return x_sim