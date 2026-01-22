import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import numdifftools as ndt

# GEV, GPD and POT utils
from .gev_utils import dq_gev, nll_gev, q_gev
# from .gpd_utils import dq_gpd, nll_gpd, q_gpd
from .pot_utils import dq_pot, q_pot, nll_pot, cdf_pot, aux_nll_pot
from .constants import LABEL_FONTSIZE, LEGEND_FONTSIZE
# from .proflike import return_proflike, return_proflike_root, return_proflike_root2
from .gev_proflikelihood import gev_rp_plik
from .gpd_profilelikelihood import gpdpoiss_rp_plik

# Optimal Threshold
from .optimal_threshold_studentized import OptimalThreshold


class ExtremeCorrection():
    """
    Extreme Correction class
    """
    def __init__(
            self,
            data_hist: pd.DataFrame,
            data_sim: pd.DataFrame,
            config: dict,
            pot_config: dict,
            method: str = "pot",
            conf_level: float = 0.95,
            random_state: int=0
    ):
        """
        Extreme value correction for sampled datasets using 
        Generalized Extreme Value (GEV) or Peaks Over Threshold (POT) approaches.

        This class applies upper-tail corrections to sampled datasets
        by fitting extreme value distributions to historical observations 
        and adjusting the sampled extremes accordingly. See V. Collado (2025) [1].

        Parameters
        ----------
        data_hist : pd.DataFrame
            Historical dataset containing the observed values.
        data_sim : pd.DataFrame
            Simulated dataset to be corrected.
        config : dict
            Dictionary containing the main configuration of the model.
            Required keys:
                - var : str
                    Variable to apply the correction.
                - time_var : str
                    Name of the time variable (datetime or timestamp).
                - yyyy_var : str
                    Name of the year variable.
                - freq : float or int
                    Frequency of observations per year 
                    (e.g., 365.25 for daily data).
            Optional keys:
                - mm_var : str, default "mm"
                    Name of the month variable.
                - dd_var : str, default "dd"
                    Name of the day variable.
                - folder : str, default None
                    Path to a folder where diagnostic plots will be saved.
        pot_config : dict
            Dictionary containing the POT configuration.
            Keys:
                - n0 : int, default 10
                    Minimum number of exceedances required.
                - min_peak_distance : int, default 2
                    Minimum distance (in data points) between two peaks.
                - init_threshold : float, default 0.0
                    Initial threshold for peak extraction.
                - siglevel : float, default 0.05
                    Significance level for the Chi-squared test in 
                    threshold optimization.
                - plot_flag : bool, default True
                    Whether to generate threshold selection plots.
        method : {"am", "pot"}, default "pot"
            Method for correction. 
            - "am" : Annual Maxima using GEV distribution.
            - "pot" : Peaks Over Threshold using GPD distribution.
        conf_level : float, default=0.95
            Confidence level for return period confidence intervals.
        random_state : int, default=0
            Random state for uniform distribution generated in extreme correction

        Attributes
        ----------
        parameters : list or None
            Distribution parameters after fitting 
            (GEV: [loc, scale, shape], POT: [threshold, scale, shape]).
        data_hist, data_sim : pd.DataFrame
            Original input datasets.
        pit_data, max_data : np.ndarray
            Point-in-time data and annual maxima extracted from historical data.
        sim_pit_data, sim_max_data : np.ndarray
            Point-in-time data and annual maxima extracted from simulated data.
        pit_data_corrected, max_data_corrected : np.ndarray
            Corrected historical data after applying the correction.
        sim_pit_data_corrected, sim_max_data_corrected : np.ndarray
            Corrected simulated data after applying the correction.

        Methods
        -------
        apply_correction(fit_diag=False)
            Fit the extreme value distribution and apply correction 
            to historical data.
        apply_sim_correction()
            Apply correction to simulated data using fitted parameters.
        extreme_fit()
            Fit the GEV or GPD distribution depending on the method.
        plot_diagnostic(save=True)
            Generate QQ and PP diagnostic plots of the fitted distribution.
        return_period_plot(show_corrected=False, show_uncorrected=True)
            Plot historical return periods with fitted distribution and CI.
        sim_return_period_plot(show_corrected=True, show_uncorrected=True)
            Plot simulated return periods with fitted distribution and CI.
        interval_return_period_plot(alpha=0.2)
            Compare return period plots across simulation intervals.
        test_dist()
            Perform goodness-of-fit test (Cramér-von Mises) 
            on fitted distributions.

        Notes
        -----
        - The correction is performed either on Annual Maxima (GEV) 
          or Peaks Over Threshold (GPD).
        - Confidence intervals of return periods are estimated using profile 
          likelihood.

        References
        ----------
        [1] V. Collado, R. Mínguez, F.J. Méndez. Upper-tail sampling correction technique for engineering design.
        https://e-archivo.uc3m.es/entities/publication/1e1a55da-f4ab-42ab-b2dd-092760946360

        """

        # Validate config dictionary
        self.config = config
        self._validate_config()

        # Define data
        self.data_hist = data_hist
        self.data_sim = data_sim

        ### Historical data
        self.max_data = self.data_hist.groupby(self.yyyy_var, as_index=False)[self.var].max()[self.var].values  # Annual Maxima
        self.max_idx = self.data_hist.groupby(self.yyyy_var)[self.var].idxmax().values                          # Annual Maxima indices
        self.max_data_sorted = np.sort(self.max_data)                                                           # Sorted Annual Maxima 
        self.pit_data = self.data_hist[self.var].values                                                         # Point-in-time data (hourly, daily...)
        self.pit_data_sorted = np.sort(self.pit_data)                                                           # Sorted point-in-time data (hourly, daily...)

        # Number of historical years and point-in-time values
        self.n_year_peaks = self.max_data.shape[0]      # Nº of years
        self.n_pit = self.pit_data.shape[0]             # Nº of point-in-time observations
        
        # Historical time interval 
        # self.time_interval_hist = self.data_hist[self.yyyy_var].max() - self.data_hist[self.yyyy_var].min()
        self.time_interval_hist = self.n_pit / self.freq   # Time interval: Number of observations / Frequency 

        ### Simulated Data
        # Annual maxima
        # self.data_sim['year'] = self.data_sim[self.time_var].dt.year
        self.sim_max_data = self.data_sim.groupby(self.yyyy_var, as_index=False)[self.var].max()[self.var].values   # Simulated annual maxima data
        self.sim_max_idx = self.data_sim.groupby(self.yyyy_var)[self.var].idxmax().values                           # Simulated annual maxima indices
        self.sim_max_data_sorted = np.sort(self.sim_max_data)                                                       # Sorted simulated annual maxima
        self.sim_pit_data = self.data_sim[self.var].values                                                          # Simulated point-in-time data
        self.sim_pit_data_sorted = np.sort(self.sim_pit_data)                                                       # Sorted simulated point-in-time data

        # Number of simulated years and point-in-time values
        self.n_sim_year_peaks = self.sim_max_data.shape[0]  # Nº of simulated years
        self.n_sim_pit = self.sim_pit_data.shape[0]         # Nº of simulated point-in-time observations

        # Simulated time interval
        # self.time_interval_sim = self.data_sim[self.yyyy_var].max() - self.data_sim[self.yyyy_var].min()
        self.time_interval_sim = self.n_sim_pit / self.freq # Time interval: Number of observations / Frequency 


        # TODO: CHECK IF THIS IS CURRENTLY NECESARY
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
        self._define_method(tolerance=0.8)
        self.method = method.lower()

        # Initilialize distribution parameters
        # If Annual Maxima (location, scale, xi); If POT (threshold, scale, xi)
        self.parameters = None  # TODO: CAMBIAR POR []?

        # Confidence level
        self.conf = conf_level

        # Set random seed
        self.random_state = random_state
        np.random.seed(self.random_state)
        

    def _validate_config(self) -> None:
        """
        Validate the configuration dictionary for extreme correction

        Raise
        -----
        KeyError
            If any required key is missing
        TypeError
            If type of any required key is wrong

        Notes
        -----
        - Required fields: "var", "time_var", "yyyy_var" and "freq"
        - Optional fields: "mm_var", "dd_var", "folder"
        """
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
        """
        Validate POT configuration dictionary for peaks extraction.
        """

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
        """
        DEPRECATED

        Select automatically the method to apply
        """

        # Obtain optimal threshold and POTs of historical data
        self.pot_data, self.pot_data_sorted, self.pot_data_locs = self.obtain_pots(
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
        self.sim_pot_data, self.sim_pot_data_sorted, self.sim_pot_data_locs = self.obtain_pots(
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
        #     self.method = 'pot'
        # else:
        #     self.method = 'am'

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

        Parameters
        ----------
        data : np.ndarray
            Data used to obtain the optimal threshold
        n0 : int, default=10
            Minimum number of exceedances required for valid computation. Defaults to 10.
        min_peak_distance : int, default=2
            Minimum distance between two peaks (in data points)
        siglevel : float, default=0.05
            Significance level for Chi-squared test
        threshold : float, default=0.0
            Initial threshold. Defaults to 0.0.
        plot_flag : bool, default=True
            Boolean flag to make plots

        Returns
        -------
        pot : np.ndarray
            Peaks Over Threshold
        pot_sorted : np.array
            Peaks Over Threshold sorted

        Attributes
        ----------
        self.opt_threshold : float
            Optimal threshold
        
        Notes
        -----
        - Optimal threshold stored in self.opt_threshold
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
        locs = opt_thres.locs

        return pot, pot_sorted, locs

    def apply_correction(
            self,
            fit_diag: bool = False
    ):
        """
        Apply correction in historical dataset

        Parameters
        ----------
        fit_diag : bool, default=False
            Whether to save the fitted diagnostic plots (PP and QQ-plots)
        """
        self.parameters = self.extreme_fit()

        if self.folder is not None and fit_diag:
            self.plot_diagnostic(save=True)

        if self.method == "pot":
            self._pot_correction()
        elif self.method == "am":
            self._annmax_correction()

    def extreme_fit(self):
        """
        Extreme fit depending on the selected extremes

        Notes
        -----
        - If self.method="pot", apply to POT
        - If self.method="am", apply to AM
        """

        if self.method == "pot":
            shape_gpd, loc_gpd, scale_gpd = stats.genpareto.fit(self.pot_data-self.opt_threshold, floc = 0)
            return [self.opt_threshold, scale_gpd, shape_gpd]

            # If the correction is applied with the Exponential
            # shape_gpd, loc_gpd, scale_gpd = stats.genpareto.fit(self.pit_data, floc = 0, fc=0)
            # loc_expon, scale_expon = stats.expon.fit(self.pot_data, floc = 0)
            # return [loc_gpd, scale_gpd, shape_gpd]
            # return [loc_expon, scale_expon, 0.0]
        
        elif self.method == "am":
            shape_gev, loc_gev, scale_gev = stats.genextreme.fit(self.max_data, 0)
            return [loc_gev, scale_gev, shape_gev]
        
        else:
            raise ValueError("The method is not selected")
        
    def plot_diagnostic(
            self,
            save: bool = True
    ):
        """
        Diagnostic plot for extreme fitting

        Includes:
        - QQplot
        - PPplot
        """
        if self.method == "pot":
            self.gpd_diag(save=save)
        
        elif self.method == "am":
            self.gev_diag(save=save)
        
        else:
            raise ValueError("The method is not selected")

    def gev_diag(self, save=True):
        """
        Diagnostic plots for GEV fitted to AM
        """
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
        """
        GEV QQ-plot
        """
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
        plt.grid()

        return fig

    def gev_ppplot(self):
        """
        GEV PP-plot
        """
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
        """
        Diagnostic plots for GPD fitted to POT
        """
        # QQ plot
        # fig = self.gpd_qqplot()
        # if save:
        #     if self.folder:  # Ensure folder is specified
        #         plt.savefig(f"{self.folder}/QQPlot.png", dpi=300, bbox_inches='tight')
        #     else:
        #         print("Warning: No folder path specified in config. Saving skipped.")
        # plt.close(fig)

        # # PP plot
        # fig = self.gpd_ppplot()
        # if save:
        #     if self.folder:
        #         plt.savefig(f"{self.folder}/PPPlot.png", dpi=300, bbox_inches='tight')
        #     else:
        #         print("Warning: No folder path specified in config. Saving skipped.")
        # plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 12))
    
        # QQ plot (top)
        probabilities_qq = (np.arange(1, self.n_pot + 1)) / (self.n_pot+1)
        gpd_quantiles = stats.genpareto.ppf(probabilities_qq, c=self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])
        
        ax1.scatter(gpd_quantiles, self.pot_data_sorted, label="Data vs GPD", alpha=0.7)
        ax1.plot(gpd_quantiles, gpd_quantiles, 'r--', label="y = x (Reference)")
        ax1.set_xlabel("Theoretical Quantiles (Fitted GPD)", fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel("Empirical Quantiles (Data)", fontsize=LABEL_FONTSIZE)
        ax1.set_title("QQ-plot", fontsize=LABEL_FONTSIZE)
        ax1.grid()

        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.tick_params(axis='both', which='minor', labelsize=18)
        
        # PP plot (bottom)
        probabilities_pp = (np.arange(1, self.n_pot + 1) - 0.5) / (self.n_pot+1)
        gpd_probs = stats.genpareto.cdf(self.pot_data_sorted, c=self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])
        
        ax2.scatter(gpd_probs, probabilities_pp, label="Empirical vs GPD", alpha=0.7)
        ax2.plot([0, 1], [0, 1], 'r--', label="y = x (Reference)")
        ax2.set_xlabel("Theoretical Probabilities (GPD)", fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel("Empirical Probabilities", fontsize=LABEL_FONTSIZE)
        ax2.set_title("PP-plot", fontsize=LABEL_FONTSIZE)
        ax2.grid()

        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.tick_params(axis='both', which='minor', labelsize=18)
        
        plt.tight_layout()
        
        if save:
            if self.folder:
                plt.savefig(f"{self.folder}/DiagnosticPlots.png", dpi=300, bbox_inches='tight')
            else:
                print("Warning: No folder path specified in config. Saving skipped.")
        
        plt.close(fig)
    
    def gpd_qqplot(self):
        """
        GPD QQ-plot
        """
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
        """
        GPD PP-plot
        """
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
            self
    ):
        """
        Apply POT extreme-correction
        """  
        # POT correction on historical data
        self.ecdf_pot_probs_hist = np.arange(1, self.n_year_peaks + 1) / (self.n_year_peaks + 1)   # ECDF
        self.runif_pot_probs_hist = np.sort(np.random.uniform(low=0, high=1, size=self.n_year_peaks))   # Random Uniform
        # self.ecdf_pot_probs_hist = np.arange(1, self.n_pot + 1) / (self.n_pot + 1)   # ECDF
        # self.runif_pot_probs_hist = np.sort(np.random.uniform(low=0, high=1, size=self.n_pot))   # Random Uniform

        # Correct in the AM using the POT model (GPD+Poisson)
        self.max_data_corrected = q_pot(self.runif_pot_probs_hist, self.parameters, self.poiss_parameter)
        # self.pot_data_corrected = stats.genpareto.ppf(                               # Corrected POTs
        #     self.runif_pot_probs_hist,
        #     self.parameters[2],
        #     loc=self.parameters[0],
        #     scale=self.parameters[1]
        # )

        # Copy point-in-time data
        # aux_pit_corrected = self.pit_data.copy()

        if self.n_year_peaks > 1:
        # if self.n_pot > 1:
            
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
                np.append(min(self.pit_data), self.max_data_sorted),   # x-coords of data points
                np.append(min(self.pit_data), self.max_data_corrected) # y-coords of data points
            )
            # aux_pit_corrected = np.interp(
            #     self.pit_data,           # x-coords to interpolate
            #     np.append(min(self.pit_data), self.pot_data_sorted),   # x-coords of data points
            #     np.append(min(self.pit_data), self.pot_data_corrected) # y-coords of data points
            # )

            # Store corrected data
            self.pit_data_corrected = aux_pit_corrected
            # self.max_data_corrected = aux_pit_corrected[self.max_idx]
            self.max_data_corrected_sort = np.sort(self.max_data_corrected)
        
        else:
            # Store corrected data
            self.pit_data_corrected = aux_pit_corrected
            self.max_data_corrected = aux_pit_corrected[self.max_idx]
            self.max_data_corrected_sort = np.sort(self.max_data_corrected)

            Warning("Only 1 POT used in the historical correction")

    def _annmax_correction(
            self
    ):
        """
        Apply AM extreme-correction
        """  
        # AnnualMaxima correction on historical data
        self.ecdf_annmax_probs_hist = np.arange(1, self.n_year_peaks + 1) / (self.n_year_peaks + 1) # ECDF
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
            show_uncorrected=True,
            conf_int_method: str="bootstrap"
    ):
        """
        Return period plot

        Parameters
        ----------
        show_corrected : bool, default=False
            If True, show the corrected AM
        show_uncorrected : bool, default=True
            If True, show the uncorrected AM
        """
        if self.method == "pot":
            fig, ax = self._pot_return_period_plot(
                show_corrected=show_corrected,
                show_uncorrected=show_uncorrected,
                conf_int_method=conf_int_method
            )
        elif self.method == "am":
            fig, ax = self._annmax_return_period_plot(
                show_corrected=show_corrected,
                show_uncorrected=show_uncorrected,
                conf_int_method=conf_int_method
            )
        return fig, ax

    def _pot_ci_return_period_bootstrap(self, B: int=1000):
        """
        Compute the Confidence intervals for return periods of GPD-Poisson based on Bootstrap method.

        Parameters
        ----------
        B : int, default=1000
            Number of bootstrap samples.
        """

        self.ci_T_years = np.array([1.001, 1.01, 1.1, 1.2, 1.4, 1.6, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 500, 1000, 10000])
        probs_ci = 1 - 1 / self.ci_T_years # Convert to exceedance probabilities

        # Generate all bootstrap samples at once
        boot_samples = np.random.choice(self.pot_data, size=(B, self.n_pot), replace=True)
        
        # Vectorized parameter fitting
        boot_params = np.zeros((B, 3))
        for i in range(B):
            shape, _, scale = stats.genpareto.fit(boot_samples[i] - self.opt_threshold, floc=0)
            boot_params[i] = [self.opt_threshold, scale, shape]

        # Vectorized return period computation
        return_periods = np.array([q_pot(probs_ci, params, self.poiss_parameter) for params in boot_params])

        self.lower_pot_ci_return = np.quantile(return_periods, (1 - self.conf) / 2, axis=0)
        self.upper_pot_ci_return = np.quantile(return_periods, 1 - (1 - self.conf) / 2, axis=0)

    def _pot_ci_return_period_proflik(self):
        """
        Compute the Confidence intervals for return periods of GPD-Poisson 
        using the Profile-Likelihood method.
        """
        self.ci_T_years = np.array([1.001, 1.01, 1.1, 1.2, 1.4, 1.6, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 500, 1000])
        # self.ci_T_years = np.array([1.1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 50, 100, 500, 1000, 5000, 10000])
        # probs = 1 - 1 / self.ci_T_years  # Convert to exceedance probabilities

        # Optimal Negative Loglikelihood
        nll_opt = nll_pot(self.pit_data, n_years=self.n_year_peaks, p=[self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2]])

        params = [self.parameters[0], self.parameters[1], self.parameters[2], self.poiss_parameter]
        lower_pot_ci_return = np.zeros_like(self.ci_T_years)
        upper_pot_ci_return = np.zeros_like(self.ci_T_years)
        
        for idx, year in enumerate(self.ci_T_years):
            # Initial bounds for confidence interval based on return period
            # xlow = min(0.1, -year//10)    # Decrease lower bound for larger return periods
            xlow = 5
            xup = max(500, year * 5)    # Increase upper bound for larger return periods
            xup = 40 + year
            
            try:
                conf_int = gpdpoiss_rp_plik(params, nll_opt, self.pit_data, m=year, 
                                        n_years=self.n_year_peaks, 
                                        xlow=xlow, xup=xup,
                                        nint=2000, conf=self.conf, 
                                        plot=True, save_file=f"{year}")
                lower_pot_ci_return[idx] = conf_int[0]
                upper_pot_ci_return[idx] = conf_int[1]
            except ValueError:
                # If bounds were insufficient, try with wider bounds
                xup = xup * 2
                xlow = xlow - xlow/2
                try:
                    conf_int = gpdpoiss_rp_plik(params, nll_opt, self.pit_data, m=year, 
                                            n_years=self.n_year_peaks, 
                                            xlow=xlow, xup=xup, 
                                            nint=2000, conf=self.conf, 
                                            plot=True, save_file=f"{year}")
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
            show_uncorrected=True,
            conf_int_method: str = "bootstrap"
    ):
        """
        Return period of Anual Maxima using the GPD-Poisson model

        Parameters
        ----------
        show_corrected : bool, default=False
            If True, show the corrected AM
        show_uncorrected : bool, default=True
            If True, show the uncorrected AM
        """

        
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
        if conf_int_method == "bootstrap":
            self._pot_ci_return_period_bootstrap()
        else:
            self._pot_ci_return_period_proflik()

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
        fig, ax = plt.subplots(figsize=(12,8))
        # ax= fig.add_subplot()
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
        ax.semilogx(self.ci_T_years, self.upper_pot_ci_return, color = "black", linestyle='dotted', linewidth=2.5, label=f'{self.conf} Conf. Band')
        ax.semilogx(self.ci_T_years, self.lower_pot_ci_return, color = "black", linestyle='dotted', linewidth=2.5)

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
            ax.semilogx(self.T_annmax, self.max_data_sorted, color="tab:blue", linewidth=0, marker='^',markersize=8, label='Historical Annual Maxima')


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
        # if self.folder is not None:
        #     plt.savefig(f"{self.folder}/Historical_ReturnPeriod.png", dpi=300, bbox_inches='tight')
        # plt.close(fig)

        return fig, ax

    def _annmax_return_period_plot(
            self, 
            show_corrected=False, 
            show_uncorrected=True,
            conf_int_method: str = "bootstrap"
            ):
        """
        Return period of Anual Maxima using the GEV

        Parameters
        ----------
        show_corrected : bool, default=False
            If True, show the corrected AM
        show_uncorrected : bool, default=True
            If True, show the uncorrected AM
        """

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
        # plt.close(fig)

        return ax

    def apply_sim_correction(
            self,
            random_state=None
    ):
        """
        Apply extreme correction procedure in sampled data
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        if self.method == "pot":
            self._pot_correction_sim()
        elif self.method == "am":
            self._annmax_correction_sim()

    def _pot_correction_sim(self):
        """
        Apply POT extreme correction procedure in sampled data
        """

        ### Apply Correction  in POTs 
        # POT 
        # self.ecdf_pot_probs_sim = np.arange(1, self.n_pot_sim + 1) / (self.n_pot_sim + 1)   # ECDF
        # self.runif_pot_probs_sim = np.sort(np.random.uniform(low=0, high=1, size=self.n_pot_sim))   # Random Uniform
        # self.sim_pot_data_corrected = stats.genpareto.ppf(self.runif_pot_probs_sim, self.parameters[2], loc=self.parameters[0], scale=self.parameters[1])    # Corrected POT
        
        # Correct in AM using POT model (GPD+Poisson)
        self.am_index_0 = 0
        for idx, value in enumerate(self.sim_max_data_sorted):
            if value == 0 or value < self.opt_threshold:
                self.am_index_0 +=1
            else:
                break

        self.sim_max_data_corrected = np.zeros(self.n_sim_year_peaks)
        for i in range(self.am_index_0):
            self.sim_max_data_corrected[i] = self.sim_max_data_sorted[i]

        # self.ecdf_max_probs_sim = np.arange(1, self.n_sim_year_peaks + 1) / (self.n_sim_year_peaks + 1)   # ECDF
        # self.runif_max_probs_sim = np.sort(np.random.uniform(low=0, high=1, size=self.n_sim_year_peaks - self.am_index_0))   # Random Uniform
        self.runif_max_probs_sim = np.sort(np.random.uniform(low=0, high=1, size=self.n_sim_year_peaks))   # Random Uniform
        self.sim_max_data_corrected[self.am_index_0:] = q_pot(self.runif_max_probs_sim[self.am_index_0:], self.parameters, self.poiss_parameter)    # Corrected POT
        # for i in range(self.n_sim_year_peaks):
        #     if self.sim_max_data_sorted[i] == 0:
        #         self.sim_max_data_corrected[i] = 0
        
        # If the correction is applied with exponential
        # self.sim_pot_data_corrected = stats.expon.ppf(self.ecdf_pot_probs_sim, loc=0.0, scale=self.parameters[1])    # Corrected POT
        
        # Annual Maxima
        # self.ecdf_annmax_probs_sim = np.arange(1, self.n_sim_peaks + 1) / (self.n_sim_peaks + 1)    # Empirical distribution function for Annual Maxima
        # Correct Annual Maxima using the fitted GEV
        # self.sim_max_data_corrected = stats.genextreme.ppf(self.ecdf_annmax_probs_sim, self.gpd_parameters[2], loc=self.gpd_parameters[0], scale=self.gpd_parameters[1])
        
        # Correct point-in-time data 
        # sim_aux_pit_corrected = self.sim_pit_data.copy()  # Copy original array
        
        # if self.n_pot_sim > 1:
        if self.n_sim_year_peaks > 1:
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
                np.append(min(self.sim_pit_data), self.sim_max_data_sorted),    # x-coords of data points
                np.append(min(self.sim_pit_data), self.sim_max_data_corrected)  # y-coords of data points 
            )
            # sim_aux_pit_corrected = np.interp(
            #     self.sim_pit_data,              # x-coords to interpolate
            #     np.append(min(self.sim_pit_data), self.sim_pot_data_sorted),    # x-coords of data points
            #     np.append(min(self.sim_pit_data), self.sim_pot_data_corrected)  # y-coords of data points 
            # )
            
            # Store the corrected data
            self.sim_pit_data_corrected = sim_aux_pit_corrected
            # self.sim_max_data_corrected = sim_aux_pit_corrected[self.sim_max_idx]
            self.sim_max_data_corrected_sorted = np.sort(self.sim_max_data_corrected)
        else:
            Warning("Not enough sampled POTs to apply the correction")
            # Store the corrected data 
            self.sim_pit_data_corrected = sim_aux_pit_corrected
            self.sim_max_data_corrected = sim_aux_pit_corrected[self.sim_max_idx]
            self.sim_max_data_corrected_sorted = np.sort(self.sim_max_data_corrected)

    def _annmax_correction_sim(self):
        """
        Apply AM extreme correction procedure in sampled data
        """
        # Correction           
        # Empirical distribution function for Annual Maxima
        self.ecdf_annmax_probs_sim = np.arange(1, self.n_sim_year_peaks + 1) / (self.n_sim_year_peaks + 1)  # ECDF
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
        """
        Plot return periods of corrected in sampled data

        Parameters
        ----------
        show_corrected : bool, default=False
            If True, show the corrected AM
        show_uncorrected : bool, default=True
            If True, show the uncorrected AM
        """
        
        if self.method == "pot":
            fig, ax = self._pot_sim_return_period_plot(
                show_corrected=show_corrected,
                show_uncorrected=show_uncorrected
            )
        elif self.method == "am":
            fig, ax = self._annmax_sim_return_period_plot(
                show_corrected=show_corrected,
                show_uncorrected=show_uncorrected
            )
        return fig, ax
    
    def _pot_sim_return_period_plot(
            self, 
            show_corrected=True, 
            show_uncorrected=True
        ):
        """
        Return period plot of AM using GPD-Poisson model

        Parameters
        ----------
        show_corrected : bool, default=False
            If True, show the corrected AM
        show_uncorrected : bool, default=True
            If True, show the uncorrected AM
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
        # self.ecdf_annmax_probs_sim = np.arange(1, self.n_sim_year_peaks + 1 - self.am_index_0) / (self.n_sim_year_peaks + 1 - self.am_index_0)
        self.T_annmax_sim = 1 / (1-self.ecdf_annmax_probs_sim) 

        # GPD-Poisson fit over a grid of x-values
        self.x_vals_gpd_poiss_sim = np.linspace(self.sim_max_data_corrected_sorted[0], self.sim_max_data_corrected_sorted[-1], 1000)
        # Return period from GPD-Poisson fit
        gpd_poiss_probs_fitted_sim = cdf_pot(self.x_vals_gpd_poiss_sim, self.parameters[0], self.poiss_parameter, self.parameters[1], self.parameters[2])
        self.T_gpd_poiss_fitted_sim = 1.0 / (1.0 - gpd_poiss_probs_fitted_sim)

        self.x_vals_gpd_poiss_sim = q_pot(1 - 1 / self.ci_T_years, self.parameters, self.poiss_parameter)

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
        ax.semilogx(self.ci_T_years, self.x_vals_gpd_poiss_sim, color = 'red',linestyle='dashed', linewidth=2.5, label='Fitted GPD-Poisson')
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
            
            # ax.semilogx(self.T_annmax_sim, self.sim_max_data_corrected_sorted[self.am_index_0:], color = 'red',linewidth=0, marker='^',markersize=5, label=r'Corrected Annual Maxima')
            ax.semilogx(self.T_annmax_sim, self.sim_max_data_corrected_sorted, color = 'red',linewidth=0, marker='^',markersize=5, label=r'Corrected Annual Maxima')
            label = label+"_Corr"
        
        # No corrected data
        if show_uncorrected:
            # ax.semilogx(T_pt_corrected_sim, self.sim_pit_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=10, fillstyle='none',markerfacecolor='none', markeredgecolor = "tab:blue", label='Daily Data')
            # ax.semilogx(T_pot_sim, self.sim_pot_data_sorted, color="orange", linewidth=0, marker='o',markersize=5, label='POT')
            
            # ax.semilogx(self.T_annmax_sim, self.sim_max_data_sorted[self.am_index_0:], color="tab:blue", linewidth=0, marker='^',markersize=5, label='Annual Maxima')
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
        ax.set_ylim(0, 100)
        ax.legend(loc='best', fontsize=LEGEND_FONTSIZE)
        ax.grid()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/Simulation_ReturnPeriod{label}.png", dpi=300, bbox_inches='tight')
        # plt.close(fig)

        return fig, ax

    def _annmax_sim_return_period_plot(
            self, 
            show_corrected=True, 
            show_uncorrected=True
        ):
        """
        Return period plot of AM using GEV model

        Parameters
        ----------
        show_corrected : bool, default=False
            If True, show the corrected AM
        show_uncorrected : bool, default=True
            If True, show the uncorrected AM
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
        # plt.close(fig)
        return fig, ax

    def interval_return_period_plot(
            self,
            alpha=0.2
    ):
        """DEPRECATED"""
        
        if self.method == "pot":
            self._pot_interval_return_period_plot(alpha=alpha)
        elif self.method == "am":
            self._annmax_interval_return_period_plot(alpha=alpha)

    def _pot_interval_return_period_plot(
            self,
            alpha=0.2
    ):
        """
        DEPRECATED
        
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
        DEPRECATED

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
        """
        Statistic test to check the goodness-of-fit of the extreme model

        Notes
        -----
        - Use the Cramér-Von Mises test
        """

        if self.method == "am":
            res_test = stats.cramervonmises(self.sim_max_data, 
                                            cdf=stats.genextreme.cdf,
                                            args=(self.parameters[2], self.parameters[0], self.parameters[1])
                                            )
            return {
                "Statistic": res_test.statistic, 
                "P-value": res_test.pvalue
                }
        
        elif self.method == "pot":
            gev_location = self.parameters[0] + (self.parameters[1] * (1 - self.poiss_parameter ** self.parameters[2])) / self.parameters[2]
            gev_scale = self.parameters[1] * self.poiss_parameter ** self.parameters[2]

            # POT test
            # res_test = stats.cramervonmises(self.sim_pot_data, 
            #                                 cdf=stats.genpareto.cdf,
            #                                 args=(self.parameters[2], self.parameters[0], self.parameters[1])
            #                                 )

            # AM test to derived GEV from GPD-Poisson
            res_test = stats.cramervonmises(self.sim_max_data, 
                                            cdf=stats.genextreme.cdf,
                                            args=(self.parameters[2], gev_location, gev_scale)
                                            )
            return {
                "Statistic": res_test.statistic, 
                "P-value": res_test.pvalue
                }

    def correlation(self):
        """
        Correlation between sampled and corrected sampled data

        Returns
        -------
        dict :
            Dictionary with Spearman, Kendall and Pearson correlation coefficients.
            Keys :
            - "Spearman" : Spearman correlation coefficient
            - "Kendall" : Kendall correlation coefficient
            - "Pearson" : Pearson correlation coefficient
        """

        spearman_corr, _ = stats.spearmanr(self.sim_pit_data, self.sim_pit_data_corrected)
        kendall_corr, _ = stats.kendalltau(self.sim_pit_data, self.sim_pit_data_corrected)
        pearson_corr, _ = stats.pearsonr(self.sim_pit_data, self.sim_pit_data_corrected)

        return {
            "Spearman": spearman_corr,
            "Kendall": kendall_corr,
            "Pearson": pearson_corr
        }
