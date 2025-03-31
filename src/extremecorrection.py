import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import numdifftools as ndt

# GEV, GPD and POT utils
from gev_utils import dq_gev, nll_gev
from gpd_utils import dq_gpd, nll_gpd
from pot_utils import dq_pot, q_pot, nll_pot, cdf_pot

# Optimal Threshold
from src.optimal_threshold_studentized import OptimalThreshold


class ExtremeCorrection():

    def __init__(
            self,
            data_hist: pd.DataFrame,
            data_sim: pd.DataFrame,
            config: dict,
            pot_config: dict,
            conf_level: float = 0.95
    ):

        # Validate config dictionary
        self.config = config
        self._validate_config()

        # Define data
        self.data_hist = data_hist
        self.data_sim = data_sim

        ### Historical data
        # Define historical data
        self.max_data = self.data_hist.groupby([self.yyyy_var], as_index=False)[self.var].max()[self.var].values    # Annual Maxima
        self.max_data_sorted = np.sort(self.max_data)                                                               # Sorted Annual Maxima 
        self.max_idx = self.data_hist.groupby([self.yyyy_var])[self.var].idxmax().values                            # Annual Maxima indices
        self.pit_data = self.data_hist[self.var].values                                                             # Point-in-time data (hourly, daily...)
        self.pit_data_sorted = np.sort(self.pit_data)                                                               # Sorted point-in-time data (hourly, daily...)

        self.time_hist = self.data_hist[self.time_var].values   # pd.Datetime variable
        self.n_year_peaks = self.max_data.shape[0]      # Nº of years
        self.n_pit = self.pit_data.shape[0]             # Nº of point-in-time observations

        ### Simulated Data
        # Annual maxima
        self.sim_max_data = self.data_sim.groupby([self.yyyy_var], as_index=False)[self.var].max()[self.var].values     # Simulated annual maxima data
        self.sim_max_idx = self.data_sim.groupby([self.yyyy_var])[self.var].idxmax().values                             # Simulated annual maxima indices
        self.sim_max_data_sorted = np.sort(self.sim_max_data)                                                           # Sorted simulated annual maxima
        self.sim_pit_data = self.data_sim[self.var].values                                                              # Simulated point-in-time data
        self.sim_pit_data_sorted = np.sort(self.sim_pit_data)                                                           # Sorted simulated point-in-time data

        self.time_sim = self.data_sim[self.time_var].values     # pd.Datetime of simulated data
        self.n_sim_year_peaks = self.sim_max_data.shape[0]   # Nº of simulated years
        self.n_sim_pit = self.sim_pit_data.shape[0]          # Nº of simulated point-in-time observations




        # POT extracting config and fit
        self.pot_config = pot_config
        self._validate_pot_config()

        # Choose the method GEV or GPD
        self._define_method()


        

    def _validate_config(self) -> None:

        # Required fields
        required_fields = {
            "var": str,
            "time_var": str,
            "yyyy_var": str,
            "freq": float | int
        }

        # Validate required fields
        for key, expected_type in required_fields.items():
            if key not in self.config:
                raise KeyError(f"Configuration error: Key '{key}' is missing in the config dictionary.")
            if not isinstance(self.config[key], expected_type):
                raise TypeError(f"Configuration error: Key '{key}' must be of type {expected_type.__name__}.")
            
    
        # Optional fields with defaults
        optional_fields = {
            "mm_var": "mm", 
            "dd_var": "dd",
            "folder": None
        }

        for key, default_value in optional_fields.items():
            self.config[key] = self.config.get(key, default_value)
        

        # Define the configuration in the class
        self.var = self.config["var"]
        self.time_var = self.config["time_var"]
        self.yyyy_var = self.config["yyyy_var"]
        self.mm_var = self.config["mm_var"]
        self.dd_var = self.config["dd_var"]
        self.freq = self.config["freq"]

        if self.config["folder"] is not None:
            self.folder = self.config["folder"]
            os.makedirs(self.folder, exist_ok=True)
        else:
            self.folder = None

    def _validate_pot_config(self) -> None:

        if self.pot_config.get('n0') is None:
            self.pot_config['n0'] = 10
        
        if self.pot_config.get('min_peak_distance') is None:
            self.pot_config['min_peak_distance'] = 2
        
        if self.pot_config.get('init_threshold') is None:
            self.pot_config['init_threshold'] = 0.0
        
        if self.pot_config.get('siglevel') is None:
            self.pot_config['siglevel'] = 0.05
        
        if self.pot_config.get('plot_flag') is None:
            self.pot_config['plot_flag'] = True

    def _define_method(self) -> None:

        # Obtain optimal threshold and POTs of historical data
        self.pot_data, self.pot_data_sorted = self.obtain_pots(
            self.pit_data,
            n0 = self.pot_config['n0'], 
            min_peak_distance = self.pot_config['min_peak_distance'], 
            siglevel = self.pot_config['siglevel'],
            threshold = self.pot_config['init_threshold'],
            plot_flag = self.pot_config['plot_flag'],
            optimize_threshold=True
        )  

        self.n_pot = self.pot_data.size                         # Nº POTs
        self.poiss_parameter = self.n_pot / self.n_year_peaks   # Poisson parameter of historical dataset

        # POT of simulated data
        self.sim_pot_data, self.sim_pot_data_sorted = self.obtain_pots(
            self.sim_pit_data,
            threshold=self.opt_threshold,
            n0 = self.pot_config['n0'],
            min_peak_distance=self.pot_config['min_peak_distance'],
            siglevel = self.pot_config['siglevel'],
            plot_flag = self.pot_config['plot_flag'],
            optimize_threshold=False
        )

        self.n_pot_sim = self.sim_pot_data.size
        self.sim_poiss_parameter = self.n_pot_sim / self.n_sim_year_peaks

    def obtain_pots(
            self, 
            data: np.ndarray,  
            n0: int=10, 
            min_peak_distance: int=2, 
            siglevel: float=0.05,
            threshold = 0.0,
            optimize_threshold = True,
            plot_flag = True
    ):
        """
        Compute the optimal threshold and the associated POTs

        Args:
            n0 (int, optional): Minimum number of exceedances required for valid computation. Defaults to 10.
            min_peak_distance (int, optional): Minimum distance between two peaks (in data points). Defaults to 2.
            siglevel (float, optional): Significance level for Chi-squared test. Defaults to 0.05.
            threshold (float, optional): Initial threshold. Defaults to 0.0.
            plot_flag (bool, optional): Boolean flag to make plots. Defaults to True.

        Returns:
            opt_threshold: Optimal threshold selected
            POTs (np.array): POTs
        """
        opt_thres = OptimalThreshold(data)

        # Peaks extraction
        opt_thres.threshold_peak_extraction(
            threshold=threshold,
            n0=n0,
            min_peak_distance=min_peak_distance
        )


        # Optimal threshold
        if optimize_threshold:
            os.makedirs(f"{self.folder}/OptimalThresholdPlots", exist_ok=True)
            self.opt_threshold = opt_thres.threshold_studentized_residuals(
                siglevel=siglevel,
                plot_flag=plot_flag,
                filename=f"{self.folder}/OptimalThresholdPlots/{self.var}",
                display_flag=False
                ).item()
        
        pot = opt_thres.pks
        pot_sorted = np.sort(pot)

        return pot, pot_sorted
        
        
