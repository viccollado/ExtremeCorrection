import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import numdifftools as ndt

# GEV, GPD and POT utils
from .gev_utils import dq_gev, nll_gev, q_gev
from .gpd_utils import dq_gpd, nll_gpd, q_gpd
from .pot_utils import dq_pot, q_pot, nll_pot, cdf_pot, aux_nll_pot
from .constants import LABEL_FONTSIZE, LEGEND_FONTSIZE
from .proflike import return_proflike, return_proflike_root, return_proflike_root2
from .gev_proflikelihood import gev_rp_plik
from .gpd_profilelikelihood import gpdpoiss_rp_plik

# Optimal Threshold
from src.optimal_threshold_studentized import OptimalThreshold


class ExtremeCorrection():

    def __init__(
            self,
            data_hist: pd.DataFrame,
            data_sim: pd.DataFrame,
            config: dict,
            pot_config: dict,
            method: str = None,
            conf_level: float = 0.95,
            tolerance: float = None
    ):

        # Validate config dictionary
        self.config = config
        self._validate_config()

        # Define data
        self.data_hist = data_hist
        self.data_sim = data_sim

        ### Historical data
        # Define historical data
        # self.data_hist['year'] = self.data_hist[self.time_var].dt.year
        self.max_data = self.data_hist.groupby(self.yyyy_var, as_index=False)[self.var].max()[self.var].values    # Annual Maxima
        self.max_idx = self.data_hist.groupby(self.yyyy_var)[self.var].idxmax().values                            # Annual Maxima indices
        self.max_data_sorted = np.sort(self.max_data)                                                               # Sorted Annual Maxima 
        self.pit_data = self.data_hist[self.var].values                                                             # Point-in-time data (hourly, daily...)
        self.pit_data_sorted = np.sort(self.pit_data)                                                               # Sorted point-in-time data (hourly, daily...)

        # self.time_hist = self.data_hist[self.time_var].values   # pd.Datetime variable
        self.time_interval_hist = self.data_hist[self.yyyy_var].max() - self.data_hist[self.yyyy_var].min()

        self.n_year_peaks = self.max_data.shape[0]      # Nº of years
        self.n_pit = self.pit_data.shape[0]             # Nº of point-in-time observations

        ### Simulated Data
        # Annual maxima
        # self.data_sim['year'] = self.data_sim[self.time_var].dt.year
        self.sim_max_data = self.data_sim.groupby(self.yyyy_var, as_index=False)[self.var].max()[self.var].values     # Simulated annual maxima data
        self.sim_max_idx = self.data_sim.groupby(self.yyyy_var)[self.var].idxmax().values                             # Simulated annual maxima indices
        self.sim_max_data_sorted = np.sort(self.sim_max_data)                                                           # Sorted simulated annual maxima
        self.sim_pit_data = self.data_sim[self.var].values                                                              # Simulated point-in-time data
        self.sim_pit_data_sorted = np.sort(self.sim_pit_data)                                                           # Sorted simulated point-in-time data

        # self.time_sim = self.data_sim[self.time_var].values     # pd.Datetime of simulated data
        self.time_interval_sim = self.data_sim[self.yyyy_var].max() - self.data_sim[self.yyyy_var].min()

        self.n_sim_year_peaks = self.sim_max_data.shape[0]   # Nº of simulated years
        self.n_sim_pit = self.sim_pit_data.shape[0]          # Nº of simulated point-in-time observations

        # Divide data in intervals of nº of historical years
        self.sim_first_year = np.min(self.data_sim[self.yyyy_var])     # First year of the simulation
        self.n_year_intervals = self.n_sim_year_peaks//self.n_year_peaks    # Nº of intervals to divide the simulated data
        self.sim_max_data_idx_intervals = {}    # Annual maximas per intervals
        for i_year in range(self.n_year_intervals):
            self.sim_max_data_idx_intervals[i_year] = self.data_sim[(self.sim_first_year + self.n_year_peaks*i_year <= self.data_sim[self.yyyy_var]) & (self.data_sim[self.yyyy_var] < self.sim_first_year+self.n_year_peaks*(i_year+1))].groupby(self.yyyy_var)[self.var].idxmax().values           


        # POT extracting config and fit
        self.pot_config = pot_config
        self._validate_pot_config()

        # Choose the method GEV or GPD
        self._define_method(tolerance=tolerance)
        self.method = method

        # Initializa distribution parameters
        # If Annual Maxima (location, scale, xi); If POT (threshold, scale, xi)
        self.parameters = None

        # Confidence level
        self.conf = conf_level

        

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

    def _define_method(
            self,
            tolerance: float = None
    ) -> None:

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
        # If the correction is applied with exponential
        # self.opt_threshold = 0.0
        # self.pot_data = self.pit_data[self.pit_data > 0]
        # self.pot_data_sorted = np.sort(self.pot_data)

        self.n_pot = self.pot_data.size                                 # Nº POTs
        self.poiss_parameter = self.n_pot / self.time_interval_hist     # Poisson parameter of historical dataset

        if tolerance is None:
            tolerance = self.poiss_parameter/100

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
        # If the correction is applied with exponential
        # self.sim_pot_data = self.sim_pit_data[self.sim_pit_data > 0]
        # self.sim_pot_data_sorted = np.sort(self.sim_pot_data)

        self.n_pot_sim = self.sim_pot_data.size                             # Nº simulated POTs
        self.sim_poiss_parameter = self.n_pot_sim / self.time_interval_sim  # Poisson parameter of historical dataset

        poiss_diff = np.abs(self.sim_poiss_parameter - self.poiss_parameter)
        # if poiss_diff < tolerance:
        #     self.method = 'POT'
        # else:
        #     self.method = 'AnnMax'

        print(f"Poisson parameters difference: {poiss_diff}")

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
            
            opt_thres.threshold_peak_extraction(
                    threshold=self.opt_threshold,
                    n0=n0,
                    min_peak_distance=min_peak_distance
                )
        
        pot = opt_thres.pks
        pot_sorted = np.sort(pot)

        return pot, pot_sorted

    def apply_correction(
            self,
            fit_diag: bool = False,
            random_state = 0
    ):
        self.parameters = self.extreme_fit()

        if self.folder is not None and fit_diag:
            self.plot_diagnostic(save=True)

        if self.method == "POT":
            self._pot_correction(random_state)
        elif self.method == "AnnMax":
            self._annmax_correction(random_state)

    def extreme_fit(self):

        if self.method == "POT":
            shape_gpd, loc_gpd, scale_gpd = stats.genpareto.fit(self.pot_data-self.opt_threshold, floc = 0)
            return [self.opt_threshold, scale_gpd, shape_gpd]

            # If the correction is applied with the Exponential
            # shape_gpd, loc_gpd, scale_gpd = stats.genpareto.fit(self.pit_data, floc = 0, fc=0)
            # loc_expon, scale_expon = stats.expon.fit(self.pot_data, floc = 0)
            # return [loc_gpd, scale_gpd, shape_gpd]
            # return [loc_expon, scale_expon, 0.0]
        
        elif self.method == "AnnMax":
            shape_gev, loc_gev, scale_gev = stats.genextreme.fit(self.max_data, 0)
            return [loc_gev, scale_gev, shape_gev]
        
        else:
            raise ValueError("The method is not selected")
        
    def plot_diagnostic(
            self,
            save: bool = True
    ):
        if self.method == "POT":
            self.gpd_diag(save=save)
        
        elif self.method == "AnnMax":
            self.gev_diag(save=save)
        
        else:
            raise ValueError("The method is not selected")

    def gev_diag(self, save=True):
        
        # QQ plot
        fig = self.gev_qqplot()
        if save:
            if self.folder:  # Ensure folder is specified
                plt.savefig(f"{self.folder}/QQPlot.png", dpi=300, bbox_inches='tight')
            else:
                print("Warning: No folder path specified in config. Saving skipped.")
        plt.close(fig)

        # PP plot
        fig = self.gev_ppplot()
        if save:
            if self.folder:
                plt.savefig(f"{self.folder}/PPPlot.png", dpi=300, bbox_inches='tight')
            else:
                print("Warning: No folder path specified in config. Saving skipped.")
        plt.close(fig)
    
    def gev_qqplot(self):

        # Calcular cuantiles teóricos de la GPD ajustada
        probabilities = (np.arange(1, self.n_year_peaks + 1)) / (self.n_year_peaks+1)  # Probabilidades de los cuantiles empíricos
        gev_quantiles = stats.genextreme.ppf(probabilities, c=self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])

        # Crear el QQ-plot
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(gev_quantiles, self.max_data_sorted, label="Data vs GEV", alpha=0.7)
        plt.plot(gev_quantiles, gev_quantiles, 'r--', label="y = x (Reference)")

        # Etiquetas
        plt.xlabel("Theoretical Quantiles (Fitted GEV)", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Empirical Quantiles (Data)", fontsize=LABEL_FONTSIZE)
        # plt.title("QQ-plot: Ajuste de la GEV a los Datos")
        # plt.legend()
        plt.grid()

        return fig

    def gev_ppplot(self):

        # Calcular cuantiles teóricos de la GPD ajustada
        probabilities = (np.arange(1, self.n_year_peaks + 1)) / (self.n_year_peaks+1)  # Probabilidades de los cuantiles empíricos
        gev_probs = stats.genextreme.cdf(self.max_data_sorted, c=self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])

        # Crear el QQ-plot
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(gev_probs, probabilities, label="Empirical vs GEV", alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--', label="y = x (Reference)")  # Reference line

        # Etiquetas
        plt.xlabel("Theoretical Probabilities (GEV)", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Empirical Probabilities", fontsize=LABEL_FONTSIZE)
        # plt.title("PP-plot: Ajuste de la GEV a los Datos")
        # plt.legend()
        plt.grid()

        return fig
    
    def gpd_diag(self, save=True):
        
        # QQ plot
        fig = self.gpd_qqplot()
        if save:
            if self.folder:  # Ensure folder is specified
                plt.savefig(f"{self.folder}/QQPlot.png", dpi=300, bbox_inches='tight')
            else:
                print("Warning: No folder path specified in config. Saving skipped.")
        plt.close(fig)

        # PP plot
        fig = self.gpd_ppplot()
        if save:
            if self.folder:
                plt.savefig(f"{self.folder}/PPPlot.png", dpi=300, bbox_inches='tight')
            else:
                print("Warning: No folder path specified in config. Saving skipped.")
        plt.close(fig)
    
    def gpd_qqplot(self):

        # Calcular cuantiles teóricos de la GPD ajustada
        probabilities = (np.arange(1, self.n_pot + 1)) / (self.n_pot+1)  # Probabilidades de los cuantiles empíricos
        gpd_quantiles = stats.genpareto.ppf(probabilities, c=self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])

        # Crear el QQ-plot
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(gpd_quantiles, self.pot_data_sorted, label="Data vs GPD", alpha=0.7)
        plt.plot(gpd_quantiles, gpd_quantiles, 'r--', label="y = x (Reference)")

        # Etiquetas
        plt.xlabel("Theoretical Quantiles (Fitted GPD)", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Empirical Quantiles (Data)", fontsize=LABEL_FONTSIZE)
        # plt.title("QQ-plot: Ajuste de la GEV a los Datos")
        # plt.legend()
        plt.grid()

        return fig

    def gpd_ppplot(self):

        # Calcular cuantiles teóricos de la GPD ajustada
        probabilities = (np.arange(1, self.n_pot + 1) - 0.5) / (self.n_pot+1)  # Probabilidades de los cuantiles empíricos
        gpd_probs = stats.genpareto.cdf(self.pot_data_sorted, c=self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])

        # Crear el QQ-plot
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(gpd_probs, probabilities, label="Empírico vs GPD", alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--', label="y = x (Referencia)")  # Reference line

        # Etiquetas
        plt.xlabel("Theoretical Probabilities (GPD)", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Empirical Probabilities", fontsize=LABEL_FONTSIZE)
        # plt.title("PP-plot: Ajuste de la GEV a los Datos")
        # plt.legend()
        plt.grid()

        return fig
    
    def _pot_correction(
            self,
            random_state
    ):
        
        # POT correction on historical data
        self.ecdf_pot_probs_hist = np.arange(1, self.n_pot + 1) / (self.n_pot + 1)   # ECDF
        np.random.seed(random_state) # Set the random seed
        self.runif_pot_probs_hist = np.sort(np.random.uniform(low=0, high=1, size=self.n_pot))   # Random Uniform

        self.pot_data_corrected = stats.genpareto.ppf(                               # Corrected POTs
            self.runif_pot_probs_hist,
            self.parameters[2],
            loc=self.parameters[0],
            scale=self.parameters[1]
        )

        # Copy point-in-time data
        # aux_pit_corrected = self.pit_data.copy()

        if self.n_pot > 1:
            
            # # Mask to interpolate
            # mask = aux_pit_corrected > self.pot_data_sorted[0]
            # # Clip values to interpolate
            # clipped_vals = np.clip(
            #     aux_pit_corrected[mask],
            #     self.pot_data_sorted[0],
            #     self.pot_data_sorted[-1]
            # )

            # # Interpolate in the peak range
            # aux_pit_corrected[mask] = np.interp(
            #     clipped_vals,           # x-coords to interpolate
            #     self.pot_data_sorted,   # x-coords of data points
            #     self.pot_data_corrected # y-coords of data points
            # )
            
            # Interpolate in the peak range
            aux_pit_corrected = np.interp(
                self.pit_data,           # x-coords to interpolate
                np.append(min(self.pit_data), self.pot_data_sorted),   # x-coords of data points
                np.append(min(self.pit_data), self.pot_data_corrected) # y-coords of data points
            )

            # Store corrected data
            self.pit_data_corrected = aux_pit_corrected
            self.max_data_corrected = aux_pit_corrected[self.max_idx]
            self.max_data_corrected_sort = np.sort(self.max_data_corrected)
        
        else:
            # Store corrected data
            self.pit_data_corrected = aux_pit_corrected
            self.max_data_corrected = aux_pit_corrected[self.max_idx]
            self.max_data_corrected_sort = np.sort(self.max_data_corrected)

            Warning("Only 1 POT used in the historical correction")

    def _annmax_correction(
            self,
            random_state
    ):

        # AnnualMaxima correction on historical data
        self.ecdf_annmax_probs_hist = np.arange(1, self.n_year_peaks + 1) / (self.n_year_peaks + 1) # ECDF
        np.random.seed(random_state) # Set the random seed
        self.runif_annmax_probs_hist = np.sort(np.random.uniform(low=0, high=1, size=self.n_year_peaks))   # Random Uniform
        self.max_data_corrected = stats.genextreme.ppf(                                      # Corrected Annual Maxima
            self.runif_annmax_probs_hist,
            self.parameters[2],
            loc=self.parameters[0],
            scale=self.parameters[1]
        )

        # Copy point-in-time data
        # aux_pit_corrected = self.pit_data.copy()

        if self.n_year_peaks > 1:
            
            # # Mask to interpolate
            # mask = aux_pit_corrected > self.max_data_sorted[0]
            # # Clip values to interpolate
            # clipped_vals = np.clip(
            #     aux_pit_corrected[mask],
            #     self.max_data_sorted[0],
            #     self.max_data_sorted[-1]
            # )

            # # Interpolate in the peak range
            # aux_pit_corrected[mask] = np.interp(
            #     clipped_vals,           # x-coords to interpolate
            #     self.max_data_sorted,   # x-coords of data points
            #     self.max_data_corrected # y-coords of data points
            # )
            
            # Interpolate in the peak range
            aux_pit_corrected = np.interp(
                self.pit_data,           # x-coords to interpolate
                np.append(min(self.pit_data), self.max_data_sorted),   # x-coords of data points
                np.append(min(self.pit_data), self.max_data_corrected) # y-coords of data points
            )

            # Store corrected data
            self.pit_data_corrected = aux_pit_corrected
            self.max_data_corrected_sort = np.sort(self.max_data_corrected)
        
        else:
            # Store corrected data
            self.pit_data_corrected = aux_pit_corrected
            self.max_data_corrected_sort = np.sort(self.max_data_corrected)

            Warning("Only 1 Annual Maxima used in the historical correction")
  
    def return_period_plot(
            self,
            show_corrected=False,
            show_uncorrected=True
    ):
        if self.method == "POT":
            self._pot_return_period_plot(
                show_corrected=show_corrected,
                show_uncorrected=show_uncorrected
            )
        elif self.method == "AnnMax":
            self._annmax_return_period_plot(
                show_corrected=show_corrected,
                show_uncorrected=show_uncorrected
            )


    def _pot_ci_return_period(self):
        # self.ci_T_years = np.array([1.1, 1.5, 2, 5, 7.5, 10, 20, 35, 50, 75, 100, 200, 500, 1000, 5000, 10000])
        self.ci_T_years = np.array([1.1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000])
        # probs = 1 - 1 / self.ci_T_years  # Convert to exceedance probabilities

        # Optimal Negative Loglikelihood
        nll_opt = nll_pot(self.pit_data, n_years=self.n_year_peaks, p=[self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2]])

        params = [self.parameters[0], self.parameters[1], self.parameters[2], self.poiss_parameter]
        lower_pot_ci_return = np.zeros_like(self.ci_T_years)
        upper_pot_ci_return = np.zeros_like(self.ci_T_years)
        
        for idx, year in enumerate(self.ci_T_years):
            # Initial bounds for confidence interval based on return period
            xlow = min(0, -year//10)    # Decrease lower bound for larger return periods
            xup = max(500, year * 5)    # Increase upper bound for larger return periods
            
            try:
                conf_int = gpdpoiss_rp_plik(params, nll_opt, self.pit_data, m=year, 
                                        n_years=self.n_year_peaks, 
                                        xlow=xlow, xup=xup,
                                        plot=False, nint=500, conf=self.conf)
                lower_pot_ci_return[idx] = conf_int[0]
                upper_pot_ci_return[idx] = conf_int[1]
            except ValueError:
                # If bounds were insufficient, try with wider bounds
                xup = xup * 2
                try:
                    conf_int = gpdpoiss_rp_plik(params, nll_opt, self.pit_data, m=year, 
                                            n_years=self.n_year_peaks, 
                                            xlow=xlow, xup=xup, 
                                            plot=False, nint=500, conf=self.conf)
                    lower_pot_ci_return[idx] = conf_int[0]
                    upper_pot_ci_return[idx] = conf_int[1]
                except ValueError as e:
                    print(f"Warning: Could not find confidence interval for return period {year}. Error: {str(e)}")
                    # Set to NaN if we can't find valid bounds
                    lower_pot_ci_return[idx] = np.nan
                    upper_pot_ci_return[idx] = np.nan

        # # GPD -----------------------------------------------
        # def nll_gpd_fixed(p, data):  # p = [sigma, shape]
        #     return nll_gpd(data, [self.parameters[0], p[0], p[1]])  # threshold fixed
        
        # gpd_ll_max_value = - nll_gpd(self.pot_data, self.parameters)

        # def q_gpd_fixed(prob, p):  # p = [sigma, shape]
        #     return q_gpd(prob, [self.parameters[0], p[0], p[1]])

        # lower_gpd_ci_return, upper_gpd_ci_return = return_proflike_root2(
        #     nll_fun=nll_gpd_fixed,
        #     init_params=self.parameters[1:],  # [sigma, shape]
        #     data=self.pot_data,
        #     quantile_fun=q_gpd_fixed,
        #     probs=probs,
        #     ll_max_value=gpd_ll_max_value,
        #     dist='gpd',
        #     sig_level=1 - self.conf
        # )

        # # POT -----------------------------------------------
        # def nll_pot_fixed(p, data):  # p = [scale, shape]
        #     return nll_pot(data, self.n_year_peaks, [self.parameters[0], self.poiss_parameter, p[0], p[1]])
        
        # pot_ll_max_value = -nll_pot(self.pit_data, self.n_year_peaks, [self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2]])

        # def q_pot_fixed(prob, p):  # p = [scale, shape]
        #     return q_pot(prob, [self.parameters[0], p[0], p[1]], self.poiss_parameter)

        # lower_pot_ci_return, upper_pot_ci_return = return_proflike_root2(
        #     nll_fun=nll_pot_fixed,
        #     init_params=self.parameters[1:],  # [scale, shape]
        #     data=self.pit_data,
        #     quantile_fun=q_pot_fixed,
        #     probs=probs,
        #     ll_max_value=pot_ll_max_value,
        #     dist='pot',
        #     sig_level=1 - self.conf
        # )

        # Store results
        # self.lower_gpd_ci_return = lower_gpd_ci_return
        # self.upper_gpd_ci_return = upper_gpd_ci_return
        self.lower_pot_ci_return = lower_pot_ci_return
        self.upper_pot_ci_return = upper_pot_ci_return
        
    def _pot_return_period_plot(
            self, 
            show_corrected=False, 
            show_uncorrected=True
    ):
        
        """ ONLY PLOT AM USING GPD-POISSON MODEL
        ###### POTs ###### 
        # GPD fit over a grid of x-values
        self.x_vals_gpd_hist = np.linspace(self.pot_data_corrected[0], self.pot_data_corrected[-1], 1000)
        # self.x_vals_gpd_hist = np.linspace(self.pot_data_corrected[0], 10000, 1000)
        
        # Return period from GPD fit
        gpd_probs_fitted = stats.genpareto.cdf(
            self.x_vals_gpd_hist, 
            self.parameters[2], 
            loc=self.parameters[0], 
            scale=self.parameters[1]
        )
        self.T_gpd_fitted = 1.0 / (1.0 - gpd_probs_fitted) / self.poiss_parameter

        # GPD Corrected peaks: re-check CDF and return periods
        ecdf_pot_probs_corrected_hist = stats.genpareto.cdf(
            stats.genpareto.ppf(
                self.ecdf_pot_probs_hist, 
                self.parameters[2], 
                loc=self.parameters[0], 
                scale=self.parameters[1]
            ),
            self.parameters[2], 
            loc=self.parameters[0], 
            scale=self.parameters[1]
        )
        self.T_ev_corrected_hist = 1.0 / (1.0 - ecdf_pot_probs_corrected_hist) / self.poiss_parameter #*(40/self.n_peaks)

        # POT (uncorrected)
        self.T_pot_hist = 1.0 / (1.0 - self.ecdf_pot_probs_hist) / self.poiss_parameter

        # self._pot_ci_return_period()
        # self.x_vals_gpd_hist = q_gpd(1 - 1 / self.ci_T_years, self.parameters)

        # Confidence intervals for GPD
        dqgpd = dq_gpd(gpd_probs_fitted, self.parameters[0], self.parameters[1], self.parameters[2])
        aux_fun = lambda x: nll_gpd(self.pot_data, x)
        hess = ndt.Hessian(aux_fun, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        hessians_gpd = hess([self.parameters[0], self.parameters[1], self.parameters[2]])   # Hessian matrix
        invI0_gpd = np.linalg.inv(hessians_gpd)     # Hessian inverse
        self.invI0_gpd=invI0_gpd
        stdDq_gpd = np.sqrt(np.sum((dqgpd.T@invI0_gpd) * dqgpd.T, axis=1))
        self.stdup_gpd = self.x_vals_gpd_hist + stdDq_gpd*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        self.stdlo_gpd = self.x_vals_gpd_hist - stdDq_gpd*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        """
        ###### Daily Data ######
        # Daily corrected data
        ecdf_pt_probs_corrected_hist = np.arange(1, self.n_pit + 1) / (self.n_pit + 1)
        T_pt_corrected_hist = 1.0 / (1.0 - ecdf_pt_probs_corrected_hist) / self.freq #/ n_return_period[wt] 
        
        # Compute Estimated Return Periods and Confidence Intervals
        self._pot_ci_return_period()
        self.x_vals_gpd_poiss_hist = q_pot(1 - 1 / self.ci_T_years, self.parameters, self.poiss_parameter)
        
        ###### Annual Maxima GPD-Poisson ######
        self.ecdf_annmax_probs_hist = np.arange(1, self.n_year_peaks + 1) / (self.n_year_peaks + 1)
        self.T_annmax = 1 / (1-self.ecdf_annmax_probs_hist)

        # GPD-Poisson fit over a grid of x-values
        # self.x_vals_gpd_poiss_hist = np.linspace(self.max_data_corrected_sort[0], self.max_data_corrected_sort[-1], 1000)
        # self.x_vals_gpd_poiss_hist = np.linspace(self.max_data_corrected_sort[0], 10000, 1000)
        # self.x_vals_gpd_poiss_hist =  q_pot(1 - 1 / self.ci_T_years, self.parameters, self.poiss_parameter)
        # Return period from GPD-Poisson fit
        # gpd_poiss_probs_fitted = cdf_pot(self.x_vals_gpd_poiss_hist, self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2])
        # self.T_gpd_poiss_fitted = 1.0 / (1.0 - gpd_poiss_probs_fitted)

        # GPD-Poisson Corrected peaks: re-check CDF and return periods
        ecdf_annmax_probs_corrected_hist = cdf_pot(
            q_pot(self.ecdf_annmax_probs_hist, self.parameters, self.poiss_parameter),
            self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2]
        )
        self.T_annmax_corrected_hist = 1.0 / (1.0 - ecdf_annmax_probs_corrected_hist) #*(40/self.n_peaks)
        

        # # Confidence Intervals
        # dqpot = dq_pot(gpd_poiss_probs_fitted, self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2])
        # aux_fun = lambda x: nll_pot(self.pit_data, self.n_year_peaks, x)
        # hess = ndt.Hessian(aux_fun, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        # hessians_gpd_poiss = hess([self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2]])
        # invI0_gpd_poiss = np.linalg.inv(hessians_gpd_poiss)   
        # stdDq_gpd_poiss = np.sqrt(np.sum((dqpot.T@invI0_gpd_poiss) * dqpot.T, axis=1)) 
        # self.stdup_gpd_poiss = q_pot(gpd_poiss_probs_fitted, [self.parameters[0], self.parameters[1], self.parameters[2]], self.poiss_parameter) + stdDq_gpd_poiss*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        # self.stdlo_gpd_poiss = q_pot(gpd_poiss_probs_fitted, [self.parameters[0], self.parameters[1], self.parameters[2]], self.poiss_parameter) - stdDq_gpd_poiss*stats.norm.ppf(1-(1-self.conf)/2,0,1)


        ### Gráfico
        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot()
        # DEPRECATED
        # # Fitted GPD
        # ax.semilogx(self.T_gpd_fitted, np.sort(self.x_vals_gpd_hist), color = 'orange',linestyle='dashed', linewidth=2.5, label='Fitted GPD')
        # # Confidence interval for fitted GPD
        # ax.semilogx(self.T_gpd_fitted, self.stdup_gpd, color = "tab:gray",linestyle='dotted')#, label=f'{self.conf} Conf. Band')
        # ax.semilogx(self.T_gpd_fitted, self.stdlo_gpd, color = "tab:gray",linestyle='dotted')

        # DEPRECATED
        # # Fitted GPD-Poisson
        # ax.semilogx(self.T_gpd_poiss_fitted, np.sort(self.x_vals_gpd_poiss_hist), color = 'red',linestyle='dashed', linewidth=2.5, label='Fitted GPD-Poisson')
        # # Confidence interval for fitted GPD-Poisson
        # ax.semilogx(self.T_gpd_poiss_fitted, self.stdup_gpd_poiss, color = "tab:gray",linestyle='dotted', label=f'{self.conf} Conf. Band')
        # ax.semilogx(self.T_gpd_poiss_fitted, self.stdlo_gpd_poiss, color = "tab:gray",linestyle='dotted')
        
        # Fitted GPD-Poisson
        ax.semilogx(self.ci_T_years, np.sort(self.x_vals_gpd_poiss_hist), color = 'red',linestyle='dashed', linewidth=2.5, label='Fitted GPD-Poisson')
        # Confidence interval for fitted GPD-Poisson
        ax.semilogx(self.ci_T_years, self.upper_pot_ci_return, color = "tab:gray",linestyle='dotted', label=f'{self.conf} Conf. Band')
        ax.semilogx(self.ci_T_years, self.lower_pot_ci_return, color = "tab:gray",linestyle='dotted')

        # Corrected data 
        if show_corrected:
            # ax.semilogx(T_pt_corrected_hist, np.sort(self.pit_data_corrected), linewidth=0, marker='o',markersize=3, label='Corrected Daily Data')
            # ax.semilogx(self.T_ev_corrected_hist, stats.genpareto.ppf(self.ecdf_pot_probs_hist, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1]), color = 'orange',linewidth=0, marker='o',markersize=3, label=r'Corrected POT')
            # ax.semilogx(self.T_annmax, q_pot(self.ecdf_annmax_probs_hist, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]), color = 'red',linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')
            ax.semilogx(self.T_annmax, self.max_data_corrected_sort, color = 'red',linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')

        # No corrected data
        if show_uncorrected:
            # ax.semilogx(T_pt_corrected_hist, self.pit_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=10, fillstyle='none',markerfacecolor='none', markeredgecolor = "tab:blue", label='Daily Data')
            # ax.semilogx(self.T_pot_hist, self.pot_data_sorted, color="orange", linewidth=0, marker='o',markersize=5, label='POTs')
            ax.semilogx(self.T_annmax, self.max_data_sorted, color="tab:blue", linewidth=0, marker='^',markersize=5, label='Annual Maxima')


        ax.set_xlabel("Return Periods (Years)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(f"{self.var}", fontsize=LABEL_FONTSIZE)
        # ax.set_title(f"Historical Return Period ({self.var})", fontsize=TITLE_FONTSIZE)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_xlim(right=10100)
        ax.set_xlim(left=0.5, right=self.n_year_peaks+100)
        # ax.set_ylim(0, np.max(np.concatenate([self.x_vals_gpd_poiss_hist, self.x_vals_gpd_hist, self.max_data_sorted]) +2))
        ax.set_ylim(0, np.max(np.concatenate([self.x_vals_gpd_poiss_hist, self.max_data_sorted]) +2))
        # ax.set_ylim(0, 100)
        ax.legend(loc='best',fontsize=LEGEND_FONTSIZE)
        ax.grid()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/Historical_ReturnPeriod.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _annmax_return_period_plot(
            self, 
            show_corrected=False, 
            show_uncorrected=True
            ):

        # GEV fit over a grid of x-values
        self.x_vals_gev_hist = np.linspace(self.max_data_corrected[0], self.max_data_corrected[-1], 1000)
        # Return period from GEV fit
        gev_probs_fitted = stats.genextreme.cdf(
            self.x_vals_gev_hist, 
            self.parameters[2], 
            loc=self.parameters[0], 
            scale=self.parameters[1]
        )
        self.T_gev_fitted = 1.0 / (1.0 - gev_probs_fitted)

        # Corrected peaks: re-check CDF and return periods
        ecdf_annmax_probs_corrected_hist = stats.genextreme.cdf(
            stats.genextreme.ppf(
                self.ecdf_annmax_probs_hist, 
                self.parameters[2], 
                loc=self.parameters[0], 
                scale=self.parameters[1]
            ),
            self.parameters[2], 
            loc=self.parameters[0], 
            scale=self.parameters[1]
        )
        self.T_ev_corrected_hist = 1.0 / (1.0 - ecdf_annmax_probs_corrected_hist) #*(40/self.n_peaks)
        
        # Daily corrected data
        ecdf_pt_probs_corrected_hist = np.arange(1, self.n_pit + 1) / (self.n_pit + 1)
        T_pt_corrected_hist = 1.0 / (1.0 - ecdf_pt_probs_corrected_hist) / self.freq #/ n_return_period[wt] 
        
        # POT (uncorrected)
        self.T_pot_hist = 1.0 / (1.0 - self.ecdf_annmax_probs_hist) #*(40/self.n_peaks)
        

        # Confidence intervals
        dqgev = dq_gev(gev_probs_fitted, p=[self.parameters[0], self.parameters[1], self.parameters[2]])
        aux_fun = lambda x: nll_gev(self.max_data, x)
        hess = ndt.Hessian(aux_fun, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        hessians_gev = hess([self.parameters[0], self.parameters[1], self.parameters[2]])
        invI0_gev = np.linalg.inv(hessians_gev)

        stdDq_gev = np.sqrt(np.sum((dqgev.T@invI0_gev) * dqgev.T, axis=1)) # Es lo mismo 
        self.stdup_gev = self.x_vals_gev_hist + stdDq_gev*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        self.stdlo_gev = self.x_vals_gev_hist - stdDq_gev*stats.norm.ppf(1-(1-self.conf)/2,0,1)



        # Gráfico
        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot()
        ax.semilogx(self.T_gev_fitted, np.sort(self.x_vals_gev_hist), color = 'red',linestyle='dashed', label='Fitted GEV')

        # Corrected data 
        if show_corrected:
            ax.semilogx(T_pt_corrected_hist, np.sort(self.pit_data_corrected), linewidth=0, marker='o',markersize=3, label='Corrected Daily Data')
            ax.semilogx(self.T_ev_corrected_hist, stats.genextreme.ppf(self.ecdf_annmax_probs_hist, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1]), color = 'red',linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')

        # No corrected data
        if show_uncorrected:
            ax.semilogx(self.T_pot_hist, self.max_data_sorted, color="red", linewidth=0, marker='o',markersize=3, label='Annual Maxima')
            ax.semilogx(T_pt_corrected_hist, self.pit_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=10, fillstyle='none',markerfacecolor='none', markeredgecolor = "tab:blue", label='Daily Data')


        # Confidence interval for fitted GEV
        ax.semilogx(self.T_gev_fitted, self.stdup_gev, color = "tab:gray",linestyle='dotted', label=f'{self.conf} Conf. Band')
        ax.semilogx(self.T_gev_fitted, self.stdlo_gev, color = "tab:gray",linestyle='dotted')

        ax.set_xlabel("Return Periods (Years)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(f"{self.var}", fontsize=LABEL_FONTSIZE)
        ax.set_title(f"Historical Return Period ({self.var})")
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(left=0.3,right=100)
        ax.set_ylim(bottom=4)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.grid()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/Historical_ReturnPeriod.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def apply_sim_correction(
            self,
            random_state=0
    ):
        
        if self.method == "POT":
            self._pot_correction_sim(random_state)
        elif self.method == "AnnMax":
            self._annmax_correction_sim(random_state)

    def _pot_correction_sim(self, random_state):

        ### Apply Correction  in POTs 
        # POT 
        self.ecdf_pot_probs_sim = np.arange(1, self.n_pot_sim + 1) / (self.n_pot_sim + 1)   # ECDF
        # Set the random seed
        np.random.seed(random_state)
        self.runif_pot_probs_sim = np.sort(np.random.uniform(low=0, high=1, size=self.n_pot_sim))   # Random Uniform
        self.sim_pot_data_corrected = stats.genpareto.ppf(self.runif_pot_probs_sim, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])    # Corrected POT
        # If the correction is applied with exponential
        # self.sim_pot_data_corrected = stats.expon.ppf(self.ecdf_pot_probs_sim, loc=0.0, scale=self.parameters[1])    # Corrected POT
        
        # Annual Maxima
        # self.ecdf_annmax_probs_sim = np.arange(1, self.n_sim_peaks + 1) / (self.n_sim_peaks + 1)    # Empirical distribution function for Annual Maxima
        # Correct Annual Maxima using the fitted GEV
        # self.sim_max_data_corrected = stats.genextreme.ppf(self.ecdf_annmax_probs_sim, self.gpd_parameters[2], loc=self.gpd_parameters[0], scale=self.gpd_parameters[1])
        
        # Correct point-in-time data 
        # sim_aux_pit_corrected = self.sim_pit_data.copy()  # Copy original array
        
        if self.n_pot_sim > 1:
            # # Create a boolean mask for values above the first “peak_values[0]”
            # mask = sim_aux_pit_corrected > self.sim_pot_data_sorted[0]
            # # Clip the values to interpolate
            # clipped_vals = np.clip(
            #     sim_aux_pit_corrected[mask], 
            #     self.sim_pot_data_sorted[0], 
            #     self.sim_pot_data_sorted[-1]
            # )
            
            # # Interpolate them onto the corrected peak range
            # sim_aux_pit_corrected[mask] = np.interp(
            #     clipped_vals,                   # x-coords to interpolate
            #     self.sim_pot_data_sorted,       # x-coords of data points
            #     self.sim_pot_data_corrected     # y-coords of data points
            # )

            sim_aux_pit_corrected = np.interp(
                self.sim_pit_data,              # x-coords to interpolate
                np.append(min(self.sim_pit_data), self.sim_pot_data_sorted),    # x-coords of data points
                np.append(min(self.sim_pit_data), self.sim_pot_data_corrected)  # y-coords of data points 
            )
            
            # Store the corrected data
            self.sim_pit_data_corrected = sim_aux_pit_corrected
            self.sim_max_data_corrected = sim_aux_pit_corrected[self.sim_max_idx]
            self.sim_max_data_corrected_sorted = np.sort(self.sim_max_data_corrected)
        else:
            Warning("Not enough sampled POTs to apply the correction")
            # Store the corrected data 
            self.sim_pit_data_corrected = sim_aux_pit_corrected
            self.sim_max_data_corrected = sim_aux_pit_corrected[self.sim_max_idx]
            self.sim_max_data_corrected_sorted = np.sort(self.sim_max_data_corrected)

    def _annmax_correction_sim(self, random_state):
 
        # Correction           
        # Empirical distribution function for Annual Maxima
        self.ecdf_annmax_probs_sim = np.arange(1, self.n_sim_year_peaks + 1) / (self.n_sim_year_peaks + 1)  # ECDF
        # Set the random seed
        np.random.seed(random_state)
        self.runif_annmax_probs_sim = np.sort(np.random.uniform(low=0, high=1, size=self.n_sim_year_peaks))   # Random Uniform
        # Correct Annual Maxima using the fitted GEV
        self.sim_max_data_corrected = stats.genextreme.ppf(self.runif_annmax_probs_sim, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])
        
        # Correct point-in-time data 
        # sim_aux_pit_corrected = self.sim_pit_data.copy()  # Copy original array
        
        if self.n_sim_year_peaks > 1:
            # # Create a boolean mask for values above the first “peak_values[0]”
            # mask = sim_aux_pit_corrected > self.sim_max_data_sorted[0]
            # # Clip the values to interpolate
            # clipped_vals = np.clip(
            #     sim_aux_pit_corrected[mask], 
            #     self.sim_max_data_sorted[0], 
            #     self.sim_max_data_sorted[-1]
            # )
            
            # Interpolate them onto the corrected peak range
            sim_aux_pit_corrected = np.interp(
                self.sim_pit_data,              # x-coords to interpolate
                np.append(min(self.sim_pit_data), self.sim_max_data_sorted),    # x-coords of data points
                np.append(min(self.sim_pit_data), self.sim_max_data_corrected)  # y-coords of data points 
            )
            
            # Store the corrected data
            self.sim_pit_data_corrected = sim_aux_pit_corrected
            self.sim_max_data_corrected_sorted = np.sort(self.sim_max_data_corrected)
        else:
            Warning("Not enough sampled Annual Maxima to apply the correction")
            # Store the corrected data 
            self.sim_pit_data_corrected = sim_aux_pit_corrected
            self.sim_max_data_corrected_sorted = np.sort(self.sim_max_data_corrected)

    def sim_return_period_plot(
            self,
            show_corrected = True,
            show_uncorrected = True
    ):
        
        if self.method == "POT":
            self._pot_sim_return_period_plot(
                show_corrected=show_corrected,
                show_uncorrected=show_uncorrected
            )
        elif self.method == "AnnMax":
            self._annmax_sim_return_period_plot(
                show_corrected=show_corrected,
                show_uncorrected=show_uncorrected
            )
    
    def _pot_sim_return_period_plot(
            self, 
            show_corrected=True, 
            show_uncorrected=True
        ):
        """
        Periodo de retorno de la serie simulada
        """

        """ DEPRECATED
        ###### POTs ###### 
        # GPD fit over a grid of x-values
        x_vals_gpd_sim = np.linspace(self.sim_pot_data_corrected[0], self.sim_pot_data_corrected[-1], 1000)
        # x_vals_gpd_sim = np.linspace(self.sim_pot_data_corrected[0], 10000, 1000)
        # Return period from GPD fitted
        gpd_probs_fitted = stats.genpareto.cdf(x_vals_gpd_sim, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])
        T_gpd_fitted = 1.0 / (1.0 - gpd_probs_fitted) / self.poiss_parameter #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # GPD Corrected peaks: re-check CDF and return periods
        ecdf_pot_probs_corrected_sim = stats.genpareto.cdf(
            stats.genpareto.ppf(self.ecdf_pot_probs_sim, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1]),
            self.parameters[2], loc=self.parameters[0], scale=self.parameters[1]
        )
        T_ev_corrected_sim = 1.0 / (1.0 - ecdf_pot_probs_corrected_sim) / self.poiss_parameter #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # POT (uncorrected)
        T_pot_sim = 1.0 / (1.0 - self.ecdf_pot_probs_sim) / self.poiss_parameter #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # Confidence intervals for GPD
        dqgpd_sim = dq_gpd(gpd_probs_fitted, self.parameters[0], self.parameters[1], self.parameters[2])
        aux_fun = lambda x: nll_gpd(self.pot_data, x)
        hess = ndt.Hessian(aux_fun, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        hessians_gpd_sim = hess([self.parameters[0], self.parameters[1], self.parameters[2]])
        invI0_gpd_sim = np.linalg.inv(hessians_gpd_sim)
        stdDq_gpd_sim = np.sqrt(np.sum((dqgpd_sim.T@invI0_gpd_sim) * dqgpd_sim.T, axis=1)) # Es lo mismo 
        stdup_gpd_sim = x_vals_gpd_sim + stdDq_gpd_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        stdlo_gpd_sim = x_vals_gpd_sim - stdDq_gpd_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        """
        ###### Daily Data ######
        # Daily corrected data
        ecdf_pt_probs_corrected_sim = np.arange(1, self.n_sim_pit + 1) / (self.n_sim_pit + 1)
        T_pt_corrected_sim = 1.0 / (1.0 - ecdf_pt_probs_corrected_sim) / self.freq #/ n_return_period[wt] 
        
        ###### Annual Maxima GPD-Poisson ######
        self.ecdf_annmax_probs_sim = np.arange(1, self.n_sim_year_peaks + 1) / (self.n_sim_year_peaks + 1)
        self.T_annmax_sim = 1 / (1-self.ecdf_annmax_probs_sim)

        # GPD-Poisson fit over a grid of x-values
        self.x_vals_gpd_poiss_sim = np.linspace(self.sim_max_data_corrected_sorted[0], self.sim_max_data_corrected_sorted[-1], 1000)
        # Return period from GPD-Poisson fit
        gpd_poiss_probs_fitted_sim = cdf_pot(self.x_vals_gpd_poiss_sim, self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2])
        self.T_gpd_poiss_fitted_sim = 1.0 / (1.0 - gpd_poiss_probs_fitted_sim)

        # GPD-Poisson Corrected peaks: re-check CDF and return periods
        ecdf_annmax_probs_corrected_sim = cdf_pot(
            q_pot(self.ecdf_annmax_probs_sim, self.parameters, self.poiss_parameter),
            self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2]
        )
        self.T_annmax_corrected_sim = 1.0 / (1.0 - ecdf_annmax_probs_corrected_sim) #*(40/self.n_peaks)
        
        # # Confidence Intervals
        # dqpot_sim = dq_pot(gpd_poiss_probs_fitted_sim, self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2])
        # aux_fun_sim = lambda x: nll_pot(self.pit_data, self.n_year_peaks, x)
        # hess_sim = ndt.Hessian(aux_fun_sim, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        # hessians_gpd_poiss_sim = hess_sim([self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2]])
        # invI0_gpd_poiss_sim = np.linalg.inv(hessians_gpd_poiss_sim)   
        # stdDq_gpd_poiss_sim = np.sqrt(np.sum((dqpot_sim.T@invI0_gpd_poiss_sim) * dqpot_sim.T, axis=1)) 
        # self.stdup_gpd_poiss_sim = q_pot(gpd_poiss_probs_fitted_sim, self.parameters, self.poiss_parameter) + stdDq_gpd_poiss_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        # self.stdlo_gpd_poiss_sim = q_pot(gpd_poiss_probs_fitted_sim, self.parameters, self.poiss_parameter) - stdDq_gpd_poiss_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)


        ### Plot
        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot()
        # # Fitted GPD
        # ax.semilogx(T_gpd_fitted, x_vals_gpd_sim, color = 'orange',linestyle='dashed', linewidth=2.5, label='Fitted GPD')
        # # Confidence interval for fitted GPD
        # ax.semilogx(T_gpd_fitted, stdup_gpd_sim, color = "tab:gray",linestyle='dotted')#, label=f'{self.conf} Conf. Band')
        # ax.semilogx(T_gpd_fitted, stdlo_gpd_sim, color = "tab:gray",linestyle='dotted')

        # # Fitted GPD-Poisson
        # ax.semilogx(self.T_gpd_poiss_fitted_sim, self.x_vals_gpd_poiss_sim, color = 'red',linestyle='dashed', linewidth=2.5, label='Fitted GPD-Poisson')
        # # Confidence interval for fitted GPD-Poisson
        # ax.semilogx(self.T_gpd_poiss_fitted_sim, self.stdup_gpd_poiss_sim, color = "tab:gray",linestyle='dotted', label=f'{self.conf} Conf. Band')
        # ax.semilogx(self.T_gpd_poiss_fitted_sim, self.stdlo_gpd_poiss_sim, color = "tab:gray",linestyle='dotted')
        
        # Fitted GPD-Poisson
        ax.semilogx(self.ci_T_years, self.x_vals_gpd_poiss_hist, color = 'red',linestyle='dashed', linewidth=2.5, label='Fitted GPD-Poisson')
        # Confidence interval for fitted GPD-Poisson
        ax.semilogx(self.ci_T_years, self.upper_pot_ci_return, color = "tab:gray",linestyle='dotted', label=f'{self.conf} Conf. Band')
        ax.semilogx(self.ci_T_years, self.lower_pot_ci_return, color = "tab:gray",linestyle='dotted')

        label=""
        # Corrected data 
        if show_corrected:
            # ax.semilogx(T_pt_corrected_sim, np.sort(self.sim_pit_data_corrected), linewidth=0, marker='o',markersize=10, fillstyle='none',markerfacecolor='none', markeredgecolor = "tab:blue", label='Corrected Daily Data')
            # ax.semilogx(T_ev_corrected_sim, stats.genpareto.ppf(self.runif_pot_probs_sim, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1]), 
            #             color = 'orange',linewidth=0, marker='o',markersize=5, label=f'Corrected POT')
            # ax.semilogx(self.T_annmax_sim, q_pot(self.ecdf_annmax_probs_sim, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]), color = 'red',linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')
            ax.semilogx(self.T_annmax_sim, self.sim_max_data_corrected_sorted, color = 'red',linewidth=0, marker='^',markersize=5, label=r'Corrected Annual Maxima')
            label = label+"_Corr"
        
        # No corrected data
        if show_uncorrected:
            # ax.semilogx(T_pt_corrected_sim, self.sim_pit_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=10, fillstyle='none',markerfacecolor='none', markeredgecolor = "tab:blue", label='Daily Data')
            # ax.semilogx(T_pot_sim, self.sim_pot_data_sorted, color="orange", linewidth=0, marker='o',markersize=5, label='POT')
            ax.semilogx(self.T_annmax_sim, self.sim_max_data_sorted, color="tab:blue", linewidth=0, marker='^',markersize=5, label='Annual Maxima')
            label = label+"_NoCorr"

    
        ax.set_xlabel("Return Periods (Years)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(f"{self.var}", fontsize=LABEL_FONTSIZE)
        # ax.set_title(f"Simulated Return Period ({self.var})", fontsize=TITLE_FONTSIZE)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())   
        # ax.set_xlim(right=10000)
        ax.set_xlim(left=0.9, right=self.n_sim_year_peaks+100)
        # ax.set_ylim(bottom=0)
        ax.set_ylim(0, 800)
        ax.legend(loc='best', fontsize=LEGEND_FONTSIZE)
        ax.grid()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/Simulation_ReturnPeriod{label}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _annmax_sim_return_period_plot(
            self, 
            show_corrected=True, 
            show_uncorrected=True
        ):
        """
        Periodo de retorno de la serie simulada
        """
        
        x_vals_gev_sim = np.linspace(self.sim_max_data_corrected[0], self.sim_max_data_corrected[-1], 1000)
        # Return period from GEV fitted
        gev_probs_fitted = stats.genextreme.cdf(x_vals_gev_sim, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])
        T_gev_fitted = 1.0 / (1.0 - gev_probs_fitted) #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # Corrected peaks: re-check CDF and return periods
        ecdf_annmax_probs_corrected_sim = stats.genextreme.cdf(
            stats.genextreme.ppf(self.ecdf_annmax_probs_sim, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1]),
            self.parameters[2], loc=self.parameters[0], scale=self.parameters[1]
        )
        T_ev_corrected_sim = 1.0 / (1.0 - ecdf_annmax_probs_corrected_sim) #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # Daily corrected data
        ecdf_pt_probs_corrected_sim = np.arange(1, self.n_sim_pit + 1) / (self.n_sim_pit + 1)
        T_pt_corrected_sim = 1.0 / (1.0 - ecdf_pt_probs_corrected_sim) / self.freq #/ n_return_period[wt] 
        
        # POT (uncorrected)
        T_pot_sim = 1.0 / (1.0 - self.ecdf_annmax_probs_sim) #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # Confidence intervals
        dqgev_sim = dq_gev(gev_probs_fitted, p=[self.parameters[0], self.parameters[1], self.parameters[2]])
        aux_fun = lambda x: nll_gev(self.max_data, x)
        hess = ndt.Hessian(aux_fun, step=1e-4)  # Añado el step para que no de problemas de inestabilidad
        hessians_gev_sim = hess([self.parameters[0], self.parameters[1], self.parameters[2]])
        invI0_gev_sim = np.linalg.inv(hessians_gev_sim)

        stdDq_gev_sim = np.sqrt(np.sum((dqgev_sim.T@invI0_gev_sim) * dqgev_sim.T, axis=1)) # Es lo mismo 
        stdup_gev_sim = x_vals_gev_sim + stdDq_gev_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        stdlo_gev_sim = x_vals_gev_sim - stdDq_gev_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)


        # Plot
        # Gráfico
        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot()
        ax.semilogx(T_gev_fitted, np.sort(x_vals_gev_sim), color = 'red',linestyle='dashed', linewidth=2.5, label='Fitted GEV')

        label=""
        # Corrected data 
        if show_corrected:
            ax.semilogx(T_pt_corrected_sim, np.sort(self.sim_pit_data_corrected), color="tab:blue", linewidth=0, marker='o',markersize=10, fillstyle='none',markerfacecolor='none', markeredgecolor = "tab:blue", label='Corrected Daily Data')
            ax.semilogx(T_ev_corrected_sim, stats.genextreme.ppf(self.runif_annmax_probs_sim, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1]), color = 'red',linewidth=0, marker='o',markersize=5, label=f'Corrected Annual Maxima')
            label = label+"_Corr"

        # No corrected data
        if show_uncorrected:
            ax.semilogx(T_pot_sim, self.sim_max_data_sorted, color="red", linewidth=0, marker='o',markersize=5, label='Annual Maxima')
            ax.semilogx(T_pt_corrected_sim, self.sim_pit_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=10, fillstyle='none',markerfacecolor='none', markeredgecolor = "tab:blue", label='Daily Data')
            label = label+"_NoCorr"

        # Confidence interval for fitted GEV
        ax.semilogx(T_gev_fitted, stdup_gev_sim, color = "black",linestyle='dotted', label=f'{self.conf} Conf. Band')
        ax.semilogx(T_gev_fitted, stdlo_gev_sim, color = "black",linestyle='dotted')

        ax.set_xlabel("Return Periods (Years)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(f"{self.var}", fontsize=LABEL_FONTSIZE)
        # ax.set_title(f"Simulated Return Period ({self.var})", fontsize=TITLE_FONTSIZE)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(left=0.5,right=2000)
        ax.set_ylim(bottom=3.9, top=20)
        ax.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)
        ax.grid()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/Simulation_ReturnPeriod{label}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def interval_return_period_plot(
            self,
            alpha=0.2
    ):
        
        if self.method == "POT":
            self._pot_interval_return_period_plot(alpha=alpha)
        elif self.method == "AnnMax":
            self._annmax_interval_return_period_plot(alpha=alpha)

    def _pot_interval_return_period_plot(
            self,
            alpha=0.2
    ):
        """
        Periodo de retorno de la serie simulada dividido en intervalos de longitud el nº de años históricos
        """

        # By years interval
        new_max_idx_sim_int = {}
        annual_maxima_corr_sim_int = {}
        ecdf_annual_maxima_sim_int = {}
        T_ecdf_annual_maxima_sim_int = {}
        # No corrected
        annual_maxima_nocorr_sim_int = {}
        ecdf_annual_maxima_nocorr_sim_int = {}
        T_ecdf_annual_maxima_nocorr_sim_int = {}
        for i_year in range(self.n_year_intervals):
            # Corrected
            new_max_idx_sim_int[i_year] = self.data_sim[self.var].index.get_indexer(self.sim_max_data_idx_intervals[i_year])
            annual_maxima_corr_sim_int[i_year] = self.sim_pit_data_corrected[new_max_idx_sim_int[i_year]]
            ecdf_annual_maxima_sim_int[i_year] = np.arange(1,len(annual_maxima_corr_sim_int[i_year])+1)/(len(annual_maxima_corr_sim_int[i_year])+1)
            T_ecdf_annual_maxima_sim_int[i_year] = 1/(1-ecdf_annual_maxima_sim_int[i_year])*(self.n_year_peaks/len(self.sim_max_data_idx_intervals[i_year]))  
            # No corrected
            annual_maxima_nocorr_sim_int[i_year] = self.data_sim[self.var][self.sim_max_data_idx_intervals[i_year]].values
            ecdf_annual_maxima_nocorr_sim_int[i_year] = np.arange(1,len(annual_maxima_nocorr_sim_int[i_year])+1)/(len(annual_maxima_nocorr_sim_int[i_year])+1)
            T_ecdf_annual_maxima_nocorr_sim_int[i_year] = 1/(1-ecdf_annual_maxima_nocorr_sim_int[i_year])*(self.n_year_peaks/len(self.sim_max_data_idx_intervals[i_year]))  

        # Plot
        fig = plt.figure(figsize=(16,8))
        ax1= fig.add_subplot(121)   
        ax2= fig.add_subplot(122)   

        # Serie simulada 40 por 40 años SIN CORREGIR
        max1 = []
        max1.append(np.max(annual_maxima_nocorr_sim_int[0]))
        ax1.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[0], np.sort(annual_maxima_nocorr_sim_int[0]), color = "tab:gray", alpha = alpha, label=f"No Corrected Simulated Data by {self.n_year_peaks} Years")
        for i_year in range(1,self.n_year_intervals):
            max1.append(np.max(annual_maxima_nocorr_sim_int[i_year]))
            ax1.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[i_year], np.sort(annual_maxima_nocorr_sim_int[i_year]), color = "tab:gray", alpha = alpha)

        # Serie simulada 40 por 40 años CORREGIDA
        max2 = []
        max2.append(np.max(annual_maxima_corr_sim_int[0]))
        ax2.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[0], np.sort(annual_maxima_corr_sim_int[0]), color = "tab:gray", alpha = alpha, label=f"Corrected Simulated Data by {self.n_year_peaks} Years")
        for i_year in range(1,self.n_year_intervals):
            max2.append(np.max(annual_maxima_corr_sim_int[i_year]))
            ax2.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[i_year], np.sort(annual_maxima_corr_sim_int[i_year]), color = "tab:gray", alpha = alpha)


        # Annual Return Periods
        # ax.semilogx(T_ev_corrected_hist[wt], stats.genextreme.ppf(ecdf_annmax_probs_hist[wt], shape_gev[wt], loc=loc_gev[wt], scale=scale_gev[wt]), 
        #             color = "#FF0000", linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')
        ax1.semilogx(self.T_gpd_poiss_fitted, np.sort(self.x_vals_gpd_poiss_hist), color = "red",linestyle='dashed', label=f'Adjusted GPD-Poisson')
        ax1.semilogx(self.T_annmax_corrected_hist, self.max_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=5, label='Annual Maxima')

        ax2.semilogx(self.T_gpd_poiss_fitted, np.sort(self.x_vals_gpd_poiss_hist), color = "red",linestyle='dashed', label=f'Adjusted GPD-Poisson')
        ax2.semilogx(self.T_annmax_corrected_hist, self.max_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=5, label='Annual Maxima')
        
        # Confidence intervals
        ax1.semilogx(self.T_gpd_poiss_fitted, self.stdup_gpd_poiss, color = "tab:gray",linestyle='dotted', label=f'{self.conf} Conf. Band')
        ax1.semilogx(self.T_gpd_poiss_fitted, self.stdlo_gpd_poiss, color = "tab:gray",linestyle='dotted')
        
        ax2.semilogx(self.T_gpd_poiss_fitted, self.stdup_gpd_poiss, color = "tab:gray",linestyle='dotted', label=f'{self.conf} Conf. Band')
        ax2.semilogx(self.T_gpd_poiss_fitted, self.stdlo_gpd_poiss, color = "tab:gray",linestyle='dotted')

        ax1.set_xlabel("Return Periods (Years)", fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel(f"{self.var}", fontsize=LABEL_FONTSIZE)
        ax1.set_xscale('log')
        ax1.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        leg1 = ax1.legend(loc='upper left', ncol=1, fontsize=LEGEND_FONTSIZE)
        for lh in leg1.legend_handles:
            lh.set_alpha(1)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax1.set_ylim(bottom = -0.1, top = np.max([np.max(max1), np.max(max2)])+0.1)
        ax1.set_xlim(right=self.n_year_peaks+50)
        ax1.set_ylim(bottom=0)
        ax1.set_title("No Corrected", fontsize=LABEL_FONTSIZE)


        ax2.set_xlabel("Return Periods (Years)", fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel(f"{self.var}", fontsize=LABEL_FONTSIZE)
        ax2.set_xscale('log')
        ax2.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        leg2 = ax2.legend(loc='upper left', ncol=1, fontsize=LEGEND_FONTSIZE)
        for lh in leg2.legend_handles:
            lh.set_alpha(1)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax2.set_ylim(bottom = -0.1, top = np.max([np.max(max1), np.max(max2)])+0.1)
        ax2.set_xlim(right=self.n_year_peaks+50)
        ax2.set_ylim(bottom=0)
        ax2.set_title("Corrected", fontsize=LABEL_FONTSIZE)

        fig.tight_layout()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/ComparativeIntervals_ReturnPeriod.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _annmax_interval_return_period_plot(
            self,
            alpha=0.2
    ):
        """
        Periodo de retorno de la serie simulada dividido en intervalos de longitud el nº de años históricos
        """

        # By years interval
        new_max_idx_sim_int = {}
        annual_maxima_corr_sim_int = {}
        ecdf_annual_maxima_sim_int = {}
        T_ecdf_annual_maxima_sim_int = {}
        # No corrected
        annual_maxima_nocorr_sim_int = {}
        ecdf_annual_maxima_nocorr_sim_int = {}
        T_ecdf_annual_maxima_nocorr_sim_int = {}
        for i_year in range(self.n_year_intervals):
            # Corrected
            new_max_idx_sim_int[i_year] = self.data_sim[self.var].index.get_indexer(self.sim_max_data_idx_intervals[i_year])
            annual_maxima_corr_sim_int[i_year] = self.sim_pit_data_corrected[new_max_idx_sim_int[i_year]]
            ecdf_annual_maxima_sim_int[i_year] = np.arange(1,len(annual_maxima_corr_sim_int[i_year])+1)/(len(annual_maxima_corr_sim_int[i_year])+1)
            T_ecdf_annual_maxima_sim_int[i_year] = 1/(1-ecdf_annual_maxima_sim_int[i_year])*(self.n_year_peaks/len(self.sim_max_data_idx_intervals[i_year]))  
            # No corrected
            annual_maxima_nocorr_sim_int[i_year] = self.data_sim[self.var][self.sim_max_data_idx_intervals[i_year]].values
            ecdf_annual_maxima_nocorr_sim_int[i_year] = np.arange(1,len(annual_maxima_nocorr_sim_int[i_year])+1)/(len(annual_maxima_nocorr_sim_int[i_year])+1)
            T_ecdf_annual_maxima_nocorr_sim_int[i_year] = 1/(1-ecdf_annual_maxima_nocorr_sim_int[i_year])*(self.n_year_peaks/len(self.sim_max_data_idx_intervals[i_year]))  

        # Plot
        fig = plt.figure(figsize=(16,8))
        ax1= fig.add_subplot(121)   
        ax2= fig.add_subplot(122)   

        # Serie simulada 40 por 40 años SIN CORREGIR
        max1 = []
        max1.append(np.max(annual_maxima_nocorr_sim_int[0]))
        ax1.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[0], np.sort(annual_maxima_nocorr_sim_int[0]), color = "tab:gray", alpha = alpha, label=f"No Corrected Simulated Data by {self.n_year_peaks} Years")
        for i_year in range(1,self.n_year_intervals):
            max1.append(np.max(annual_maxima_nocorr_sim_int[i_year]))
            ax1.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[i_year], np.sort(annual_maxima_nocorr_sim_int[i_year]), color = "tab:gray", alpha = alpha)

        # Serie simulada 40 por 40 años CORREGIDA
        max2 = []
        max2.append(np.max(annual_maxima_corr_sim_int[0]))
        ax2.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[0], np.sort(annual_maxima_corr_sim_int[0]), color = "tab:gray", alpha = alpha, label=f"Corrected Simulated Data by {self.n_year_peaks} Years")
        for i_year in range(1,self.n_year_intervals):
            max2.append(np.max(annual_maxima_corr_sim_int[i_year]))
            ax2.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[i_year], np.sort(annual_maxima_corr_sim_int[i_year]), color = "tab:gray", alpha = alpha)


        # Annual Return Periods
        # ax.semilogx(T_ev_corrected_hist[wt], stats.genextreme.ppf(ecdf_annmax_probs_hist[wt], shape_gev[wt], loc=loc_gev[wt], scale=scale_gev[wt]), 
        #             color = "#FF0000", linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')
        ax1.semilogx(self.T_gev_fitted, np.sort(self.x_vals_gev_hist), color = "red",linestyle='dashed', label=f'Adjusted GEV')
        ax1.semilogx(self.T_pot_hist, self.max_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=5, label='Annual Maxima')
        
        ax2.semilogx(self.T_gev_fitted, np.sort(self.x_vals_gev_hist), color = "red",linestyle='dashed', label=f'Adjusted GEV')
        ax2.semilogx(self.T_pot_hist, self.max_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=5, label='Annual Maxima')
        
        # Confidence intervals
        ax1.semilogx(self.T_gev_fitted, self.stdup_gev, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int')
        ax1.semilogx(self.T_gev_fitted, self.stdlo_gev, color = "black",linestyle='dotted')
        
        ax2.semilogx(self.T_gev_fitted, self.stdup_gev, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int')
        ax2.semilogx(self.T_gev_fitted, self.stdlo_gev, color = "black",linestyle='dotted')
        
        

        ax1.set_xlabel("Return Periods (Years)", fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel(f"{self.var}", fontsize=LABEL_FONTSIZE)
        ax1.set_xscale('log')
        ax1.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        leg1 = ax1.legend(loc='best', ncol=1, fontsize=LEGEND_FONTSIZE)
        for lh in leg1.legend_handles:
            lh.set_alpha(1)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax1.set_ylim(bottom = 4, top = np.max([np.max(max1), np.max(max2)])+0.1)
        ax1.set_xlim(right=self.n_year_peaks+10)
        ax1.set_title("No Corrected", fontsize=LABEL_FONTSIZE)


        ax2.set_xlabel("Return Periods (Years)", fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel(f"{self.var}", fontsize=LABEL_FONTSIZE)
        ax2.set_xscale('log')
        ax2.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        leg2 = ax2.legend(loc='best', ncol=1, fontsize=LEGEND_FONTSIZE)
        for lh in leg2.legend_handles:
            lh.set_alpha(1)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax2.set_ylim(bottom = 4, top = np.max([np.max(max1), np.max(max2)])+0.1)
        ax2.set_xlim(right=self.n_year_peaks+10)
        ax2.set_title("Corrected", fontsize=LABEL_FONTSIZE)

        fig.tight_layout()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/ComparativeIntervals_ReturnPeriod.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


    def test_dist(self):

        if self.method == "AnnMax":
            res_test = stats.cramervonmises(self.sim_max_data, 
                                            cdf=stats.genextreme.cdf,
                                            args=(self.parameters[2], self.parameters[0], self.parameters[1])
                                            )
            return {
                "Statistic": res_test.statistic, 
                "P-value": res_test.pvalue
                }
        
        elif self.method == "POT":
            res_test = stats.cramervonmises(self.sim_pot_data, 
                                            cdf=stats.genpareto.cdf,
                                            args=(self.parameters[2], self.parameters[0], self.parameters[1])
                                            )
            return {
                "Statistic": res_test.statistic, 
                "P-value": res_test.pvalue
                }