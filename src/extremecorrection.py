import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from gev_utils import dq_gev, nll_gev
import numdifftools as ndt



class Gev_ExtremeCorrection():
    """
    Extremal Correction of Annual Maxima
    """
    def __init__(self, 
                 data: pd.DataFrame, 
                 config: dict,
                 conf: float = 0.95

                 # Antiguas variables
                #  data_var: str = 'Hs', frequency: float | int = 365.25, 
                #  year_var: str = 'yyyy', month_var: str = 'mm'
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

        self.n_peaks = self.max_data.shape[0]
        self.n_pit = self.pit_data.shape[0]

    
        # Stationary GEV parameters (location, scale, shape)
        self.gev_parameters = None

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

    
    def apply_correction(self, fit_diag=False):
        
        # Adjust GEV to Annual Maxima
        self.gev_parameters = self.gev_fit()

        if self.folder is not None and fit_diag:
            self.gev_diag(save=True)

        ## Correction on historical data                    
        # Empirical distribution function for Annual Maxima
        self.ecdf_annmax_probs_hist = np.arange(1, self.n_peaks + 1) / (self.n_peaks + 1)
        # Correct Annual Maxima using the fitted GEV
        self.max_data_corrected = stats.genextreme.ppf(self.ecdf_annmax_probs_hist, self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1])
        
        # Correct point-in-time data 
        aux_pit_corrected = self.pit_data.copy()  # Copy original array
        
        if self.n_peaks > 1:
            # Create a boolean mask for values above the first “peak_values[0]”
            above_mask = aux_pit_corrected > self.max_data_sorted[0]
            # Clip the values to interpolate
            clipped_vals = np.clip(aux_pit_corrected[above_mask], self.max_data_sorted[0], self.max_data_sorted[-1])
            
            # Interpolate them onto the corrected peak range
            aux_pit_corrected[above_mask] = np.interp(
                clipped_vals,               # x-coords to interpolate
                self.max_data_sorted,       # x-coords of data points
                self.max_data_corrected     # y-coords of data points
            )
            
            # Store the corrected data
            self.pit_data_corrected = aux_pit_corrected
        else:
            # Store the corrected data 
            self.pit_data_corrected = aux_pit_corrected

    def return_period_plot(self, show_corrected=False, show_uncorrected=True):

        # GEV fit over a grid of x-values
        self.x_vals_gev_hist = np.linspace(self.max_data_corrected[0], self.max_data_corrected[-1], 1000)
        # Return period from GEV fit
        gev_probs_fitted = stats.genextreme.cdf(self.x_vals_gev_hist, self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1])
        self.T_gev_fitted = 1.0 / (1.0 - gev_probs_fitted)

        # Corrected peaks: re-check CDF and return periods
        ecdf_annmax_probs_corrected_hist = stats.genextreme.cdf(
            stats.genextreme.ppf(self.ecdf_annmax_probs_hist, self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1]),
            self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1]
        )
        self.T_ev_corrected_hist = 1.0 / (1.0 - ecdf_annmax_probs_corrected_hist) #*(40/self.n_peaks)
        
        # Daily corrected data
        ecdf_pt_probs_corrected_hist = np.arange(1, self.n_pit + 1) / (self.n_pit + 1)
        T_pt_corrected_hist = 1.0 / (1.0 - ecdf_pt_probs_corrected_hist) / self.freq #/ n_return_period[wt] 
        
        # POT (uncorrected)
        self.T_pot_hist = 1.0 / (1.0 - self.ecdf_annmax_probs_hist) #*(40/self.n_peaks)
        

        # Confidence intervals
        dqgev = dq_gev(self.ecdf_annmax_probs_hist, p=[self.gev_parameters[0], self.gev_parameters[1], self.gev_parameters[2]])
        aux_fun = lambda x: nll_gev(self.max_data, x)
        hess = ndt.Hessian(aux_fun, step=1e-5)  # Añado el step para que no de problemas de inestabilidad
        hessians_gev = hess([self.gev_parameters[0], self.gev_parameters[1], self.gev_parameters[2]])
        invI0_gev = np.linalg.inv(hessians_gev)

        stdDq_gev = np.sqrt(np.sum((dqgev.T@invI0_gev) * dqgev.T, axis=1)) # Es lo mismo 
        self.stdup_gev = self.max_data_corrected + stdDq_gev*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        self.stdlo_gev = self.max_data_corrected - stdDq_gev*stats.norm.ppf(1-(1-self.conf)/2,0,1)



        # Gráfico
        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot()
        ax.semilogx(self.T_gev_fitted, np.sort(self.x_vals_gev_hist), color = 'orange',linestyle='dashed', label='Fitted GEV')

        # Corrected data 
        if show_corrected:
            ax.semilogx(T_pt_corrected_hist, np.sort(self.pit_data_corrected), linewidth=0, marker='o',markersize=3, label='Corrected Daily Data')
            ax.semilogx(self.T_ev_corrected_hist, stats.genextreme.ppf(self.ecdf_annmax_probs_hist, self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1]), color = 'orange',linewidth=0, marker='o',markersize=3, label=r'Corrected Annual Maxima')

        # No corrected data
        if show_uncorrected:
            ax.semilogx(self.T_pot_hist, self.max_data_sorted, color="green", linewidth=0, marker='+',markersize=3, label='Annual Maxima')
            ax.semilogx(T_pt_corrected_hist, self.pit_data_sorted, color="purple", linewidth=0, marker='+',markersize=3, label='Daily Data')


        # Confidence interval for fitted GEV
        ax.semilogx(self.T_ev_corrected_hist, self.stdup_gev, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int')
        ax.semilogx(self.T_ev_corrected_hist, self.stdlo_gev, color = "black",linestyle='dotted')

        ax.set_xlabel("Return Periods (Years)")
        ax.set_ylabel(f"{self.var}")
        ax.set_title(f"Historical Return Period ({self.var})")
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(right=250)
        ax.legend()
        ax.grid()
        if self.folder is not None:
            plt.savefig(f"{self.folder}/Historical_ReturnPeriod.png", dpi=300)
        plt.close(fig)

    def gev_fit(self):
        shape_gev, loc_gev, scale_gev = stats.genextreme.fit(self.max_data, 0)
        return [loc_gev, scale_gev, shape_gev]
    
    def gev_diag(self, save=True):
        
        # QQ plot
        fig = self.gev_qqplot()
        if save:
            if self.folder:  # Ensure folder is specified
                plt.savefig(f"{self.folder}/QQPlot.png", dpi=300)
            else:
                print("Warning: No folder path specified in config. Saving skipped.")
        plt.close(fig)

        # PP plot
        fig = self.gev_ppplot()
        if save:
            if self.folder:
                plt.savefig(f"{self.folder}/PPPlot.png", dpi=300)
            else:
                print("Warning: No folder path specified in config. Saving skipped.")
        plt.close(fig)
    
    def gev_qqplot(self):

        # Calcular cuantiles teóricos de la GPD ajustada
        n = len(self.max_data)
        probabilities = (np.arange(1, n + 1)) / (n+1)  # Probabilidades de los cuantiles empíricos
        gev_quantiles = stats.genextreme.ppf(probabilities, c=self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1])

        # Crear el QQ-plot
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(gev_quantiles, self.max_data_sorted, label="Datos vs GEV", alpha=0.7)
        plt.plot(gev_quantiles, gev_quantiles, 'r--', label="y = x (Referencia)")

        # Etiquetas
        plt.xlabel("Cuantiles Teóricos (GEV ajustada)")
        plt.ylabel("Cuantiles Empíricos (Datos)")
        # plt.title("QQ-plot: Ajuste de la GEV a los Datos")
        # plt.legend()
        plt.grid()

        return fig

    def gev_ppplot(self):

        # Calcular cuantiles teóricos de la GPD ajustada
        probabilities = (np.arange(1, self.n_peaks + 1)) / (self.n_peaks+1)  # Probabilidades de los cuantiles empíricos
        gev_probs = stats.genextreme.cdf(self.max_data_sorted, c=self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1])

        # Crear el QQ-plot
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(probabilities, gev_probs, label="Empírico vs GEV", alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--', label="y = x (Referencia)")  # Reference line

        # Etiquetas
        plt.xlabel("Probabilidades Empíricas")
        plt.ylabel("Probabilidades Teóricas (GEV)")
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

        self.sim_max_data = simulated_data.groupby([self.yyyy_var], as_index=False)[self.var].max()[self.var].values     # Simulated annual maxima data
        self.sim_max_idx = simulated_data.groupby([self.yyyy_var])[self.var].idxmax().values                             # Simulated annual maxima indices
        self.sim_max_data_sorted = np.sort(self.sim_max_data)                                                                 # Sorted simulated annual maxima
        self.sim_max_data_sorted_idx = np.argsort(self.sim_max_data)                                                          # Indices of sorted simulated annual maxima

        self.sim_pit_data = simulated_data[self.var].values      # Simulated point-in-time data
        self.sim_pit_data_sorted = np.sort(self.sim_pit_data)         # Sorted simulated point-in-time data

        self.n_sim_peaks = self.sim_max_data.shape[0]
        self.n_sim_pit = self.sim_pit_data.shape[0]


        self.sim_first_year = np.min(simulated_data[self.yyyy_var].values)
        self.n_year_intervals = self.n_sim_peaks//self.n_peaks
        # Divide the simulated data in intervals of historical length
        self.sim_max_data_idx_intervals = {}
        for i_year in range(self.n_year_intervals):
            self.sim_max_data_idx_intervals[i_year] = simulated_data[(self.sim_first_year + self.n_peaks*i_year <= simulated_data[self.yyyy_var]) & (simulated_data[self.yyyy_var] < self.sim_first_year+self.n_peaks*(i_year+1))].groupby([self.yyyy_var])[self.var].idxmax().values           


        # Correction           
        # Empirical distribution function for Annual Maxima
        self.ecdf_annmax_probs_sim = np.arange(1, self.n_sim_peaks + 1) / (self.n_sim_peaks + 1)
        # Correct Annual Maxima using the fitted GEV
        self.sim_max_data_corrected = stats.genextreme.ppf(self.ecdf_annmax_probs_sim, self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1])
        
        # Correct point-in-time data 
        sim_aux_pit_corrected = self.sim_pit_data.copy()  # Copy original array
        
        if self.n_sim_peaks > 1:
            # Create a boolean mask for values above the first “peak_values[0]”
            above_mask = sim_aux_pit_corrected > self.sim_max_data_sorted[0]
            # Clip the values to interpolate
            clipped_vals = np.clip(sim_aux_pit_corrected[above_mask], self.sim_max_data_sorted[0], self.sim_max_data_sorted[-1])
            
            # Interpolate them onto the corrected peak range
            sim_aux_pit_corrected[above_mask] = np.interp(
                clipped_vals,              # x-coords to interpolate
                self.sim_max_data_sorted,       # x-coords of data points
                self.sim_max_data_corrected     # y-coords of data points
            )
            
            # Store the corrected data
            self.sim_pit_data_corrected = sim_aux_pit_corrected
        else:
            # Store the corrected data 
            self.sim_pit_data_corrected = sim_aux_pit_corrected


    def sim_return_period_plot(self, show_corrected=True, show_uncorrected=True):
        """
        Periodo de retorno de la serie simulada
        """
        
        x_vals_gev_sim = np.linspace(self.sim_max_data_corrected[0], self.sim_max_data_corrected[-1], 1000)
        # Return period from GEV fitted
        gev_probs_fitted = stats.genextreme.cdf(x_vals_gev_sim, self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1])
        T_gev_fitted = 1.0 / (1.0 - gev_probs_fitted) #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # Corrected peaks: re-check CDF and return periods
        ecdf_annmax_probs_corrected_sim = stats.genextreme.cdf(
            stats.genextreme.ppf(self.ecdf_annmax_probs_sim, self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1]),
            self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1]
        )
        T_ev_corrected_sim = 1.0 / (1.0 - ecdf_annmax_probs_corrected_sim) #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # Daily corrected data
        ecdf_pt_probs_corrected_sim = np.arange(1, self.n_sim_pit + 1) / (self.n_sim_pit + 1)
        T_pt_corrected_sim = 1.0 / (1.0 - ecdf_pt_probs_corrected_sim) / self.freq #/ n_return_period[wt] 
        
        # POT (uncorrected)
        T_pot_sim = 1.0 / (1.0 - self.ecdf_annmax_probs_sim) #*(40/len(max_data_hist[wt]))#(10000/n_peaks)
        
        # Confidence intervals
        dqgev_sim = dq_gev(self.ecdf_annmax_probs_sim, p=[self.gev_parameters[0], self.gev_parameters[1], self.gev_parameters[2]])
        aux_fun = lambda x: nll_gev(self.max_data, x)
        hess = ndt.Hessian(aux_fun, step=1e-4)  # Añado el step para que no de problemas de inestabilidad
        hessians_gev_sim = hess([self.gev_parameters[0], self.gev_parameters[1], self.gev_parameters[2]])
        invI0_gev_sim = np.linalg.inv(hessians_gev_sim)

        stdDq_gev_sim = np.sqrt(np.sum((dqgev_sim.T@invI0_gev_sim) * dqgev_sim.T, axis=1)) # Es lo mismo 
        stdup_gev_sim = self.sim_max_data_corrected + stdDq_gev_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)
        stdlo_gev_sim = self.sim_max_data_corrected - stdDq_gev_sim*stats.norm.ppf(1-(1-self.conf)/2,0,1)


        # Plot
        # Gráfico
        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot()
        ax.semilogx(T_gev_fitted, np.sort(x_vals_gev_sim), color = 'orange',linestyle='dashed', label='Fitted GEV')

        # Corrected data 
        if show_corrected:
            ax.semilogx(T_pt_corrected_sim, np.sort(self.sim_pit_data_corrected), linewidth=0, marker='o',markersize=3, label=f'Corrected Daily Data')
            ax.semilogx(T_ev_corrected_sim, stats.genextreme.ppf(self.ecdf_annmax_probs_sim, self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1]), color = 'orange',linewidth=0, marker='o',markersize=3, label=f'Corrected Annual Maxima')

        # No corrected data
        if show_uncorrected:
            ax.semilogx(T_pot_sim, self.sim_max_data_sorted, color="green", linewidth=0, marker='+',markersize=3, label='Annual Maxima')
            ax.semilogx(T_pt_corrected_sim, self.sim_pit_data_sorted, color="purple", linewidth=0, marker='+',markersize=3, label='Daily Data')


        # Confidence interval for fitted GEV
        ax.semilogx(T_ev_corrected_sim, stdup_gev_sim, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int')
        ax.semilogx(T_ev_corrected_sim, stdlo_gev_sim, color = "black",linestyle='dotted')

        ax.set_xlabel("Return Periods (Years)")
        ax.set_ylabel(f"{self.var}")
        ax.set_title(f"Simulated Return Period ({self.var})")
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(right=self.n_sim_peaks+self.n_sim_peaks//10)
        ax.legend()
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
            T_ecdf_annual_maxima_sim_int[i_year] = 1/(1-ecdf_annual_maxima_sim_int[i_year])*(self.n_peaks/len(self.sim_max_data_idx_intervals[i_year]))  
            # No corrected
            annual_maxima_nocorr_sim_int[i_year] = self.simulated_data[self.var][~np.isnan(self.simulated_data[self.var].values)][self.sim_max_data_idx_intervals[i_year]].values
            ecdf_annual_maxima_nocorr_sim_int[i_year] = np.arange(1,len(annual_maxima_nocorr_sim_int[i_year])+1)/(len(annual_maxima_nocorr_sim_int[i_year])+1)
            T_ecdf_annual_maxima_nocorr_sim_int[i_year] = 1/(1-ecdf_annual_maxima_nocorr_sim_int[i_year])*(self.n_peaks/len(self.sim_max_data_idx_intervals[i_year]))  

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
        ax1.semilogx(self.T_gev_fitted, np.sort(self.x_vals_gev_hist), color = "tab:red",linestyle='dashed', label=f'Adjusted GEV')
        ax1.semilogx(self.T_pot_hist, self.max_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=4, label='Annual Maxima')
        
        ax2.semilogx(self.T_gev_fitted, np.sort(self.x_vals_gev_hist), color = "tab:red",linestyle='dashed', label=f'Adjusted GEV')
        ax2.semilogx(self.T_pot_hist, self.max_data_sorted, color="tab:blue", linewidth=0, marker='o',markersize=4, label='Annual Maxima')
        
        # Confidence intervals
        ax1.semilogx(self.T_ev_corrected_hist, self.stdup_gev, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int')
        ax1.semilogx(self.T_ev_corrected_hist, self.stdlo_gev, color = "black",linestyle='dotted')
        
        ax2.semilogx(self.T_ev_corrected_hist, self.stdup_gev, color = "black",linestyle='dotted', label=f'{self.conf} Conf Int')
        ax2.semilogx(self.T_ev_corrected_hist, self.stdlo_gev, color = "black",linestyle='dotted')
        
        

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

    def time_series_plot():
        """
        TODO: Añadir esta función que dibuje la serie temporal tanto corregida como la no corregida para compararlas 
            - Incluir tambien puntos en los máximos históricos y triangulos en los simulados o algo similar para compararlos
        """
        ...
    