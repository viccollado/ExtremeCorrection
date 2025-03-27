import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import numdifftools as ndt
from typing import Dict

# GPD and POT utils
from gpd_utils import dq_gpd, nll_gpd
from pot_utils import dq_pot, q_pot, nll_pot, cdf_pot

# Optimal Threshold
from OptimalThresholdSelection.optimal_threshold_studentized import OptimalThreshold



class GPD_ExtremeCorrection():
    """
    Extremal Correction of POTs using GPD.
    """
    def __init__(self, 
                 data: pd.DataFrame, 
                 config: dict,
                 pot_config: dict,
                 conf: float = 0.95,

                 # Antiguas variables
                 # data_var: str = 'Hs', frequency: float | int = 365.25, 
                 # year_var: str = 'yyyy', month_var: str = 'mm'
                 ):
        """
        Initialize the extreme correction methodology

        Args:
            data (pd.DataFrame): Historic data including variables month ('mm') and year ('yyyy').
            config (dict): Model configuration, includes:
                - var (str): Variable to perform the analysis
                - yyyy_var (str): Name of year variable
                - mm_var (str): Optional, Name of month variable
                - dd_var (str): Optional, Name of day variable
                - freq (float): Frequency of data along one year (if daily data, freq = 365.25)
                - folder (str): Folder to save plots.
            pot_config (dict): Configuration for peaks extraction.
            conf (float): Confidence level for confidence intervals
        """

        # Validate config dictionary
        self.config = config
        self.validate_config()

        # Define data
        self.data = data

        # Definir variables (np.array)
        self.max_data = self.data.groupby([self.yyyy_var], as_index=False)[self.var].max()[self.var].values # Annual Maxima
        self.max_data_sorted = np.sort(self.max_data)                                                       # Annual Maxima sorted
        self.max_idx = self.data.groupby([self.yyyy_var])[self.var].idxmax().values                         # Annual Maxima indices
        self.pit_data = self.data[self.var].values                                                          # Point-in-time data (hourly, daily...)
        self.pit_data_sorted = np.sort(self.pit_data)                                                       # Point-in-time data (hourly, daily...)

        self.n_year_peaks = self.max_data.shape[0]
        self.n_pit = self.pit_data.shape[0]

        # POT extracting config and fit
        self.pot_config = pot_config
        self.pot_data, self.pot_data_sorted = self.obtain_pots(
            self.pit_data,
            n0 = self.pot_config['n0'], 
            min_peak_distance = self.pot_config['min_peak_distance'], 
            siglevel = self.pot_config['siglevel'],
            threshold = self.pot_config['init_threshold'],
            plot_flag = self.pot_config['plot_flag'],
            optimize_threshold=True
        )  
    
        self.n_pot_peaks = len(self.pot_data)

        # Stationary GEV parameters (location, scale, shape)
        self.gpd_parameters = None
        self.poiss_parameter = self.n_pot_peaks / self.n_year_peaks

        # Corrected data
        self.max_data_corrected = None
        self.pit_data_corrected = None

        # Confidence level
        self.conf = conf

    
    def validate_config(self):
        # Required fields
        required_fields = {
            "var": str,
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
        self.yyyy_var = self.config["yyyy_var"]
        self.mm_var = self.config["mm_var"]
        self.dd_var = self.config["dd_var"]
        self.freq = self.config["freq"]

        if self.config["folder"] is not None:
            self.folder = self.config["folder"]
            os.makedirs(self.folder, exist_ok=True)
        else:
            self.folder = None

    def validate_pot_config(self):

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
        
        pot = opt_thres.pks[opt_thres.pks > self.opt_threshold]
        pot_sorted = np.sort(pot)

        return pot, pot_sorted
        
    
    def apply_correction(self, fit_diag=False):
        
        # Adjust GEV to Annual Maxima
        self.gpd_parameters = self.gpd_fit()

        if self.folder is not None and fit_diag:
            self.gpd_diag(save=True)

        ## Correction on historical data                    
        # Empirical distribution function for Annual Maxima
        self.ecdf_pot_probs_hist = np.arange(1, self.n_pot_peaks + 1) / (self.n_pot_peaks + 1)
        # Correct Annual Maxima using the fitted GEV
        self.pot_data_corrected = stats.genpareto.ppf(self.ecdf_pot_probs_hist, self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1])
        
        # Correct point-in-time data 
        aux_pit_corrected = self.pit_data.copy()  # Copy original array
        
        if self.n_pot_peaks > 1:
            # Create a boolean mask for values above the first “peak_values[0]”
            above_mask = aux_pit_corrected > self.pot_data_sorted[0]
            # Clip the values to interpolate
            clipped_vals = np.clip(aux_pit_corrected[above_mask], self.pot_data_sorted[0], self.pot_data_sorted[-1])
            
            # Interpolate them onto the corrected peak range
            aux_pit_corrected[above_mask] = np.interp(
                clipped_vals,           # x-coords to interpolate
                self.pot_data_sorted,   # x-coords of data points
                self.pot_data_corrected # y-coords of data points
            )
            
            # Store the corrected data
            self.pit_data_corrected = aux_pit_corrected
            self.max_data_corrected = aux_pit_corrected[self.max_idx]
            self.max_data_corrected_sort = np.sort(self.max_data_corrected)
        else:
            # Store the corrected data 
            self.pit_data_corrected = aux_pit_corrected
            self.max_data_corrected = aux_pit_corrected[self.max_idx]
            self.max_data_corrected_sort = np.sort(self.max_data_corrected)

    def return_period_plot(self, show_corrected=False, show_uncorrected=True):
        

        ###### POTs ###### 
        # GPD fit over a grid of x-values
        self.x_vals_gpd_hist = np.linspace(self.pot_data_corrected[0], self.pot_data_corrected[-1], 1000)
        # Return period from GPD fit
        gpd_probs_fitted = stats.genpareto.cdf(self.x_vals_gpd_hist, self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1])
        self.T_gpd_fitted = 1.0 / (1.0 - gpd_probs_fitted) / self.poiss_parameter

        # GPD Corrected peaks: re-check CDF and return periods
        ecdf_pot_probs_corrected_hist = stats.genpareto.cdf(
            stats.genpareto.ppf(self.ecdf_pot_probs_hist, self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1]),
            self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1]
        )
        self.T_ev_corrected_hist = 1.0 / (1.0 - ecdf_pot_probs_corrected_hist) / self.poiss_parameter #*(40/self.n_peaks)

        # POT (uncorrected)
        self.T_pot_hist = 1.0 / (1.0 - self.ecdf_pot_probs_hist) / self.poiss_parameter

        # Confidence intervals for GPD
        dqgpd = dq_gpd(self.ecdf_pot_probs_hist, self.opt_threshold, self.gpd_parameters[1], self.gpd_parameters[2])
        aux_fun = lambda x: nll_gpd(self.pot_data, x)
        hess = ndt.Hessian(aux_fun, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        hessians_gpd = hess([self.opt_threshold, self.gpd_parameters[1], self.gpd_parameters[2]])   # Hessian matrix
        invI0_gpd = np.linalg.inv(hessians_gpd)     # Hessian inverse
        stdDq_gpd = np.sqrt(np.sum((dqgpd.T@invI0_gpd) * dqgpd.T, axis=1))
        self.stdup_gpd = self.pot_data_corrected + stdDq_gpd*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        self.stdlo_gpd = self.pot_data_corrected - stdDq_gpd*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        
        ###### Daily Data ######
        # Daily corrected data
        ecdf_pt_probs_corrected_hist = np.arange(1, self.n_pit + 1) / (self.n_pit + 1)
        T_pt_corrected_hist = 1.0 / (1.0 - ecdf_pt_probs_corrected_hist) / self.freq #/ n_return_period[wt] 
        

        ###### Annual Maxima GPD-Poisson ######
        self.ecdf_annmax_probs_hist = np.arange(1, self.n_year_peaks + 1) / (self.n_year_peaks + 1)
        self.T_annmax = 1 / (1-self.ecdf_annmax_probs_hist)

        # GPD-Poisson fit over a grid of x-values
        self.x_vals_gpd_poiss_hist = np.linspace(self.max_data_corrected_sort[0], self.max_data_corrected_sort[-1], 1000)
        # Return period from GPD-Poisson fit
        gpd_poiss_probs_fitted = cdf_pot(self.x_vals_gpd_poiss_hist, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2])
        self.T_gpd_poiss_fitted = 1.0 / (1.0 - gpd_poiss_probs_fitted)

        # GPD-Poisson Corrected peaks: re-check CDF and return periods
        ecdf_annmax_probs_corrected_hist = cdf_pot(
            q_pot(self.ecdf_annmax_probs_hist, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]),
            self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]
        )
        self.T_annmax_corrected_hist = 1.0 / (1.0 - ecdf_annmax_probs_corrected_hist) #*(40/self.n_peaks)

        # Confidence Intervals
        dqpot = dq_pot(self.ecdf_annmax_probs_hist, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2])
        aux_fun = lambda x: nll_pot(self.pit_data, x)
        hess = ndt.Hessian(aux_fun, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        hessians_gpd_poiss = hess([self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]])
        invI0_gpd_poiss = np.linalg.inv(hessians_gpd_poiss)   
        stdDq_gpd_poiss = np.sqrt(np.sum((dqpot.T@invI0_gpd_poiss) * dqpot.T, axis=1)) 
        self.stdup_gpd_poiss = q_pot(ecdf_annmax_probs_corrected_hist, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]) + stdDq_gpd_poiss*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        self.stdlo_gpd_poiss = q_pot(ecdf_annmax_probs_corrected_hist, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]) - stdDq_gpd_poiss*stats.norm.ppf(1-(1-self.conf)/2,0,1)


        ### Gráfico
        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot()
        # Fitted GPD
        ax.semilogx(self.T_gpd_fitted, np.sort(self.x_vals_gpd_hist), color = 'orange',linestyle='dashed', label='Fitted GPD')
        # Confidence interval for fitted GPD
        ax.semilogx(self.T_ev_corrected_hist, self.stdup_gpd, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int (GPD)')
        ax.semilogx(self.T_ev_corrected_hist, self.stdlo_gpd, color = "black",linestyle='dotted')

        # Fitted GPD-Poisson
        ax.semilogx(self.T_gpd_poiss_fitted, np.sort(self.x_vals_gpd_poiss_hist), color = 'red',linestyle='dashed', label='Fitted GPD-Poisson')
        # Confidence interval for fitted GPD-Poisson
        ax.semilogx(self.T_annmax_corrected_hist, self.stdup_gpd_poiss, color = "tab:grey",linestyle='dashdot', label=f'{self.conf} Conf Int (GPD-Poisson)')
        ax.semilogx(self.T_annmax_corrected_hist, self.stdlo_gpd_poiss, color = "tab:grey",linestyle='dashdot')

        # Corrected data 
        if show_corrected:
            ax.semilogx(T_pt_corrected_hist, np.sort(self.pit_data_corrected), linewidth=0, marker='o',markersize=3, label='Corrected Daily Data')
            ax.semilogx(self.T_ev_corrected_hist, stats.genpareto.ppf(self.ecdf_pot_probs_hist, self.gpd_parameters[2], loc=self.gpd_parameters[0], scale=self.gpd_parameters[1]), color = 'orange',linewidth=0, marker='o',markersize=3, label=r'Corrected POT')
            # ax.semilogx(self.T_annmax, q_pot(self.ecdf_annmax_probs_hist, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]), color = 'red',linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')
            ax.semilogx(self.T_annmax, self.max_data_corrected_sort, color = 'red',linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')

        # No corrected data
        if show_uncorrected:
            ax.semilogx(T_pt_corrected_hist, self.pit_data_sorted, color="purple", linewidth=0, marker='+',markersize=3, label='Daily Data')
            ax.semilogx(self.T_pot_hist, self.pot_data_sorted, color="green", linewidth=0, marker='+',markersize=3, label='POTs')
            ax.semilogx(self.T_annmax, self.max_data_sorted, color="red", linewidth=0, marker='+',markersize=3, label='Annual Maxima')


        

        ax.set_xlabel("Return Periods (Years)")
        ax.set_ylabel(f"{self.var}")
        ax.set_title(f"Historical Return Period ({self.var})")
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(right=10000)
        ax.set_ylim(0, np.max(np.concatenate([self.x_vals_gpd_poiss_hist, self.x_vals_gpd_hist]) +0.5))
        ax.legend()
        ax.grid()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/Historical_ReturnPeriod.png", dpi=300)
        plt.close(fig)

    def gpd_fit(self):
        shape_gpd, loc_gpd, scale_gpd = stats.genpareto.fit(self.pot_data-self.opt_threshold, floc = 0)
        return [loc_gpd, scale_gpd, shape_gpd]
    
    def gpd_diag(self, save=True):
        
        # QQ plot
        fig = self.gpd_qqplot()
        if save:
            if self.folder:  # Ensure folder is specified
                plt.savefig(f"{self.folder}/QQPlot.png", dpi=300)
            else:
                print("Warning: No folder path specified in config. Saving skipped.")
        plt.close(fig)

        # PP plot
        fig = self.gpd_ppplot()
        if save:
            if self.folder:
                plt.savefig(f"{self.folder}/PPPlot.png", dpi=300)
            else:
                print("Warning: No folder path specified in config. Saving skipped.")
        plt.close(fig)
    
    def gpd_qqplot(self):

        # Calcular cuantiles teóricos de la GPD ajustada
        probabilities = (np.arange(1, self.n_pot_peaks + 1)) / (self.n_pot_peaks+1)  # Probabilidades de los cuantiles empíricos
        gpd_quantiles = stats.genpareto.ppf(probabilities, c=self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1])

        # Crear el QQ-plot
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(gpd_quantiles, self.pot_data_sorted, label="Datos vs GPD", alpha=0.7)
        plt.plot(gpd_quantiles, gpd_quantiles, 'r--', label="y = x (Referencia)")

        # Etiquetas
        plt.xlabel("Cuantiles Teóricos (GPD ajustada)")
        plt.ylabel("Cuantiles Empíricos (Datos)")
        # plt.title("QQ-plot: Ajuste de la GEV a los Datos")
        # plt.legend()
        plt.grid()

        return fig

    def gpd_ppplot(self):

        # Calcular cuantiles teóricos de la GPD ajustada
        probabilities = (np.arange(1, self.n_pot_peaks + 1) - 0.5) / (self.n_pot_peaks+1)  # Probabilidades de los cuantiles empíricos
        gpd_probs = stats.genpareto.cdf(self.pot_data_sorted, c=self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1])

        # Crear el QQ-plot
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(probabilities, gpd_probs, label="Empírico vs GPD", alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--', label="y = x (Referencia)")  # Reference line

        # Etiquetas
        plt.xlabel("Probabilidades Empíricas")
        plt.ylabel("Probabilidades Teóricas (GPD)")
        # plt.title("PP-plot: Ajuste de la GEV a los Datos")
        # plt.legend()
        plt.grid()

        return fig
    

    def apply_correction_sim(self, simulated_data: pd.DataFrame):
        """_summary_

        Args:
            simulated_data (pd.DataFrame): Simulated data frame in with the same vairables as the historical one.
        """
        self.simulated_data = simulated_data

        # Annual maxima
        self.sim_max_data = simulated_data.groupby([self.yyyy_var], as_index=False)[self.var].max()[self.var].values        # Simulated annual maxima data
        self.sim_max_idx = simulated_data.groupby([self.yyyy_var])[self.var].idxmax().values                                # Simulated annual maxima indices
        self.sim_max_data_sorted = np.sort(self.sim_max_data)                                                               # Sorted simulated annual maxima
        self.sim_max_data_sorted_idx = np.argsort(self.sim_max_data)                                                        # Indices of sorted simulated annual maxima

        # Point-in-time
        self.sim_pit_data = simulated_data[self.var].values      # Simulated point-in-time data
        self.sim_pit_data_sorted = np.sort(self.sim_pit_data)         # Sorted simulated point-in-time data

        # POT 
        self.sim_pot_data, self.sim_pot_data_sorted = self.obtain_pots(
            self.sim_pit_data,
            threshold=self.opt_threshold,
            optimize_threshold=False
        )

        # Number of data (annual maxima, point-in-time and pot)
        self.n_sim_year_peaks = self.sim_max_data.shape[0]
        self.n_sim_pit = self.sim_pit_data.shape[0]
        self.n_sim_pot_peaks = self.sim_pot_data.shape[0] 

        self.sim_poiss_parameter = self.n_sim_pot_peaks/self.n_sim_year_peaks   # Poisson parameter of simulated data

        self.sim_first_year = np.min(simulated_data[self.yyyy_var].values)  # First year of the simulation
        self.n_year_intervals = self.n_sim_year_peaks//self.n_year_peaks    # Nº of intervals to divide the simulated data
        # Divide the simulated data in intervals of historical length
        self.sim_max_data_idx_intervals = {}    # Annual maximas per intervals
        for i_year in range(self.n_year_intervals):
            self.sim_max_data_idx_intervals[i_year] = simulated_data[(self.sim_first_year + self.n_year_peaks*i_year <= simulated_data[self.yyyy_var]) & (simulated_data[self.yyyy_var] < self.sim_first_year+self.n_year_peaks*(i_year+1))].groupby([self.yyyy_var])[self.var].idxmax().values           


        ### Apply Correction  in POTs 
        # POT 
        self.ecdf_pot_probs_sim = np.arange(1, self.n_sim_pot_peaks + 1) / (self.n_sim_pot_peaks + 1)   # Empirical Dist Funct
        self.sim_pot_data_corrected = stats.genpareto.ppf(self.ecdf_pot_probs_sim, self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1])    # Corrected POT
        
        # Annual Maxima
        # self.ecdf_annmax_probs_sim = np.arange(1, self.n_sim_peaks + 1) / (self.n_sim_peaks + 1)    # Empirical distribution function for Annual Maxima
        # Correct Annual Maxima using the fitted GEV
        # self.sim_max_data_corrected = stats.genextreme.ppf(self.ecdf_annmax_probs_sim, self.gpd_parameters[2], loc=self.gpd_parameters[0], scale=self.gpd_parameters[1])
        
        # Correct point-in-time data 
        sim_aux_pit_corrected = self.sim_pit_data.copy()  # Copy original array
        
        if self.n_sim_pot_peaks > 1:
            # Create a boolean mask for values above the first “peak_values[0]”
            above_mask = sim_aux_pit_corrected > self.sim_pot_data_sorted[0]
            # Clip the values to interpolate
            clipped_vals = np.clip(sim_aux_pit_corrected[above_mask], self.sim_pot_data_sorted[0], self.sim_pot_data_sorted[-1])
            
            # Interpolate them onto the corrected peak range
            sim_aux_pit_corrected[above_mask] = np.interp(
                clipped_vals,                   # x-coords to interpolate
                self.sim_pot_data_sorted,       # x-coords of data points
                self.sim_pot_data_corrected     # y-coords of data points
            )
            
            # Store the corrected data
            self.sim_pit_data_corrected = sim_aux_pit_corrected
            self.sim_max_data_corrected = sim_aux_pit_corrected[self.sim_max_idx]
            self.sim_max_data_corrected_sorted = np.sort(self.sim_max_data_corrected)
        else:
            # Store the corrected data 
            self.sim_pit_data_corrected = sim_aux_pit_corrected
            self.sim_max_data_corrected = sim_aux_pit_corrected[self.sim_max_idx]
            self.sim_max_data_corrected_sorted = np.sort(self.sim_max_data_corrected)


    def sim_return_period_plot(self, show_corrected=True, show_uncorrected=True):
        """
        Periodo de retorno de la serie simulada
        """
        
        ###### POTs ###### 
        # GPD fit over a grid of x-values
        x_vals_gpd_sim = np.linspace(self.sim_pot_data_corrected[0], self.sim_pot_data_corrected[-1], 1000)
        # Return period from GPD fitted
        gpd_probs_fitted = stats.genpareto.cdf(x_vals_gpd_sim, self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1])
        T_gpd_fitted = 1.0 / (1.0 - gpd_probs_fitted) / self.poiss_parameter #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # GPD Corrected peaks: re-check CDF and return periods
        ecdf_pot_probs_corrected_sim = stats.genpareto.cdf(
            stats.genpareto.ppf(self.ecdf_pot_probs_sim, self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1]),
            self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1]
        )
        T_ev_corrected_sim = 1.0 / (1.0 - ecdf_pot_probs_corrected_sim) / self.poiss_parameter #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # POT (uncorrected)
        T_pot_sim = 1.0 / (1.0 - self.ecdf_pot_probs_sim) / self.poiss_parameter #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # Confidence intervals for GPD
        dqgpd_sim = dq_gpd(self.ecdf_pot_probs_sim, self.opt_threshold, self.gpd_parameters[1], self.gpd_parameters[2])
        aux_fun = lambda x: nll_gpd(self.pot_data, x)
        hess = ndt.Hessian(aux_fun, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        hessians_gpd_sim = hess([self.opt_threshold, self.gpd_parameters[1], self.gpd_parameters[2]])
        invI0_gpd_sim = np.linalg.inv(hessians_gpd_sim)
        stdDq_gpd_sim = np.sqrt(np.sum((dqgpd_sim.T@invI0_gpd_sim) * dqgpd_sim.T, axis=1)) # Es lo mismo 
        stdup_gpd_sim = self.sim_pot_data_corrected + stdDq_gpd_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        stdlo_gpd_sim = self.sim_pot_data_corrected - stdDq_gpd_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)

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
        gpd_poiss_probs_fitted_sim = cdf_pot(self.x_vals_gpd_poiss_sim, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2])
        self.T_gpd_poiss_fitted_sim = 1.0 / (1.0 - gpd_poiss_probs_fitted_sim)

        # GPD-Poisson Corrected peaks: re-check CDF and return periods
        ecdf_annmax_probs_corrected_sim = cdf_pot(
            q_pot(self.ecdf_annmax_probs_sim, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]),
            self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]
        )
        self.T_annmax_corrected_sim = 1.0 / (1.0 - ecdf_annmax_probs_corrected_sim) #*(40/self.n_peaks)

        # Confidence Intervals
        dqpot_sim = dq_pot(self.ecdf_annmax_probs_sim, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2])
        aux_fun_sim = lambda x: nll_pot(self.pit_data, x)
        hess_sim = ndt.Hessian(aux_fun_sim, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        hessians_gpd_poiss_sim = hess_sim([self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]])
        invI0_gpd_poiss_sim = np.linalg.inv(hessians_gpd_poiss_sim)   
        stdDq_gpd_poiss_sim = np.sqrt(np.sum((dqpot_sim.T@invI0_gpd_poiss_sim) * dqpot_sim.T, axis=1)) 
        self.stdup_gpd_poiss_sim = q_pot(ecdf_annmax_probs_corrected_sim, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]) + stdDq_gpd_poiss_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        self.stdlo_gpd_poiss_sim = q_pot(ecdf_annmax_probs_corrected_sim, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]) - stdDq_gpd_poiss_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)


        ### Plot
        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot()
        # Fitted GPD
        ax.semilogx(T_gpd_fitted, np.sort(x_vals_gpd_sim), color = 'orange',linestyle='dashed', label='Fitted GPD')
        # Confidence Interval for fitted GPD
        ax.semilogx(T_ev_corrected_sim, stdup_gpd_sim, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int (GPD)')
        ax.semilogx(T_ev_corrected_sim, stdlo_gpd_sim, color = "black",linestyle='dotted')

        # Fitted GPD-Poisson
        ax.semilogx(self.T_gpd_poiss_fitted_sim, np.sort(self.x_vals_gpd_poiss_sim), color = 'red',linestyle='dashed', label='Fitted GPD-Poisson')
        # Confidence interval for fitted GPD-Poisson
        ax.semilogx(self.T_annmax_corrected_sim, self.stdup_gpd_poiss_sim, color = "tab:grey",linestyle='dashdot', label=f'{self.conf} Conf Int (GPD-Poisson)')
        ax.semilogx(self.T_annmax_corrected_sim, self.stdlo_gpd_poiss_sim, color = "tab:grey",linestyle='dashdot')



        # Corrected data 
        if show_corrected:
            ax.semilogx(T_pt_corrected_sim, np.sort(self.sim_pit_data_corrected), linewidth=0, marker='o',markersize=3, label=f'Corrected Daily Data')
            ax.semilogx(T_ev_corrected_sim, stats.genpareto.ppf(self.ecdf_pot_probs_sim, self.gpd_parameters[2], loc=self.opt_threshold, scale=self.gpd_parameters[1]), 
                        color = 'orange',linewidth=0, marker='o',markersize=3, label=f'Corrected POT')
            # ax.semilogx(self.T_annmax_sim, q_pot(self.ecdf_annmax_probs_sim, self.opt_threshold, self.poiss_parameter, self.gpd_parameters[1], self.gpd_parameters[2]), color = 'red',linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')
            ax.semilogx(self.T_annmax_sim, self.sim_max_data_corrected_sorted, color = 'red',linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')

        # No corrected data
        if show_uncorrected:
            ax.semilogx(T_pt_corrected_sim, self.sim_pit_data_sorted, color="purple", linewidth=0, marker='+',markersize=3, label='Daily Data')
            ax.semilogx(T_pot_sim, self.sim_pot_data_sorted, color="green", linewidth=0, marker='+',markersize=3, label='POT')
            ax.semilogx(self.T_annmax_sim, self.sim_max_data_sorted, color="red", linewidth=0, marker='+',markersize=3, label='Annual Maxima')


    
        ax.set_xlabel("Return Periods (Years)")
        ax.set_ylabel(f"{self.var}")
        ax.set_title(f"Simulated Return Period ({self.var})")
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(right=10000)
        ax.legend(loc='upper left')
        ax.grid()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/Simulation_ReturnPeriod.png", dpi=300)
        plt.close(fig)
        
    def interval_sim_return_period_plot(self):
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
            new_max_idx_sim_int[i_year] = self.simulated_data[self.var][~np.isnan(self.simulated_data[self.var].values)].index.get_indexer(self.sim_max_data_idx_intervals[i_year])
            annual_maxima_corr_sim_int[i_year] = self.sim_pit_data_corrected[new_max_idx_sim_int[i_year]]
            ecdf_annual_maxima_sim_int[i_year] = np.arange(1,len(annual_maxima_corr_sim_int[i_year])+1)/(len(annual_maxima_corr_sim_int[i_year])+1)
            T_ecdf_annual_maxima_sim_int[i_year] = 1/(1-ecdf_annual_maxima_sim_int[i_year])*(self.n_year_peaks/len(self.sim_max_data_idx_intervals[i_year]))  
            # No corrected
            annual_maxima_nocorr_sim_int[i_year] = self.simulated_data[self.var][~np.isnan(self.simulated_data[self.var].values)][self.sim_max_data_idx_intervals[i_year]].values
            ecdf_annual_maxima_nocorr_sim_int[i_year] = np.arange(1,len(annual_maxima_nocorr_sim_int[i_year])+1)/(len(annual_maxima_nocorr_sim_int[i_year])+1)
            T_ecdf_annual_maxima_nocorr_sim_int[i_year] = 1/(1-ecdf_annual_maxima_nocorr_sim_int[i_year])*(self.n_year_peaks/len(self.sim_max_data_idx_intervals[i_year]))  

        # Plot
        fig = plt.figure(figsize=(16,8))
        ax1= fig.add_subplot(121)   
        ax2= fig.add_subplot(122)   

        # Serie simulada 40 por 40 años SIN CORREGIR
        max1 = []
        max1.append(np.max(annual_maxima_nocorr_sim_int[0]))
        ax1.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[0], np.sort(annual_maxima_nocorr_sim_int[0]), color = "tab:gray", alpha = 0.1, label="No Corrected Simulated Data by 40 Years")
        for i_year in range(1,self.n_year_intervals):
            max1.append(np.max(annual_maxima_nocorr_sim_int[i_year]))
            ax1.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[i_year], np.sort(annual_maxima_nocorr_sim_int[i_year]), color = "tab:gray", alpha = 0.1)

        # Serie simulada 40 por 40 años CORREGIDA
        max2 = []
        max2.append(np.max(annual_maxima_corr_sim_int[0]))
        ax2.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[0], np.sort(annual_maxima_corr_sim_int[0]), color = "tab:gray", alpha = 0.1, label="Corrected Simulated Data by 40 Years")
        for i_year in range(1,self.n_year_intervals):
            max2.append(np.max(annual_maxima_corr_sim_int[i_year]))
            ax2.semilogx(T_ecdf_annual_maxima_nocorr_sim_int[i_year], np.sort(annual_maxima_corr_sim_int[i_year]), color = "tab:gray", alpha = 0.1)


        # Annual Return Periods
        # ax.semilogx(T_ev_corrected_hist[wt], stats.genextreme.ppf(ecdf_annmax_probs_hist[wt], shape_gev[wt], loc=loc_gev[wt], scale=scale_gev[wt]), 
        #             color = "#FF0000", linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')
        ax1.semilogx(self.T_gpd_poiss_fitted, np.sort(self.x_vals_gpd_hist), color = "tab:red",linestyle='dashed', label=f'Adjusted GEV')
        ax1.semilogx(self.T_annmax_corrected_hist, self.max_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=4, label='Annual Maxima')
        
        ax2.semilogx(self.T_gpd_poiss_fitted, np.sort(self.x_vals_gpd_hist), color = "tab:red",linestyle='dashed', label=f'Adjusted GEV')
        ax2.semilogx(self.T_annmax_corrected_hist, self.max_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=4, label='Annual Maxima')
        
        # Confidence intervals
        ax1.semilogx(self.T_annmax_corrected_hist, self.stdup_gpd_poiss, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int')
        ax1.semilogx(self.T_annmax_corrected_hist, self.stdlo_gpd_poiss, color = "black",linestyle='dotted')
        
        ax2.semilogx(self.T_annmax_corrected_hist, self.stdup_gpd_poiss, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int')
        ax2.semilogx(self.T_annmax_corrected_hist, self.stdlo_gpd_poiss, color = "black",linestyle='dotted')
        
        

        ax1.set_xlabel("Return Periods (Years)")
        ax1.set_ylabel(f"{self.var}")
        ax1.set_xscale('log')
        ax1.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        leg1 = ax1.legend(loc='upper left', ncol=2)
        for lh in leg1.legend_handles:
            lh.set_alpha(1)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax1.set_ylim(bottom = -0.1, top = np.max([np.max(max1), np.max(max2)])+0.1)
        ax1.set_xlim(right=50)
        ax1.set_title("No Corrected")


        ax2.set_xlabel("Return Periods (Years)")
        ax2.set_ylabel(f"{self.var}")
        ax2.set_xscale('log')
        ax2.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        leg2 = ax2.legend(loc='upper left', ncol=2)
        for lh in leg2.legend_handles:
            lh.set_alpha(1)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax2.set_ylim(bottom = -0.1, top = np.max([np.max(max1), np.max(max2)])+0.1)
        ax2.set_xlim(right=50)
        ax2.set_title("Corrected")

        fig.tight_layout()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/ComparativeIntervals_ReturnPeriod.png", dpi=300)
        plt.close(fig)

    def time_series_plot(self, sim=False):
        """
        Time series plot 

        Args:
            sim (bool, optional): If True Simulated TS plot, else Historical TS plot. Defaults to False.
        """

        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)

        # Historical time series
        if not sim:
            ax.plot(np.arange(1, self.n_pit+1)/self.freq, self.pit_data, label="No Corrected Historical data")
            ax.plot(np.arange(1, self.n_pit+1)/self.freq, self.pit_data_corrected, label="Corrected Historical data")
            ax.scatter((np.arange(1, self.n_pit+1)/self.freq)[self.max_idx], self.max_data, label="No Corrected Annual Maxima")
            ax.scatter((np.arange(1, self.n_pit+1)/self.freq)[self.max_idx], self.pit_data_corrected[self.max_idx], label="Corrected Annual Maxima")
            ax.set_xticks(np.arange(1, self.n_year_peaks+1))

        # Simulated time series
        if sim:
            ax.plot(np.arange(1, self.n_sim_pit+1)/self.freq, self.sim_pit_data, label="No Corrected Simulated data")
            ax.plot(np.arange(1, self.n_sim_pit+1)/self.freq, self.sim_pit_data_corrected, label="Corrected Simulated data")
            ax.scatter((np.arange(1, self.n_sim_pit+1)/self.freq)[self.sim_max_idx], self.sim_max_data, label="No Corrected Annual Maxima")
            ax.scatter((np.arange(1, self.n_sim_pit+1)/self.freq)[self.sim_max_idx], self.sim_pit_data_corrected[self.sim_max_idx], label="Corrected Annual Maxima")
            ax.set_xticks(np.arange(1, self.n_sim_year_peaks+1))
        
        ax.legend()
        ax.grid()
        fig.tight_layout()
        if self.folder is not None:
            if not sim:
                plt.savefig(f"{self.folder}/TimeSeries_hist.png", dpi=300)
            if sim:
                plt.savefig(f"{self.folder}/TimeSeries_sim.png", dpi=300)
        plt.close(fig)