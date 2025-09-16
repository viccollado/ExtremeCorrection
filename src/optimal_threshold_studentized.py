import numpy as np
import matplotlib.pyplot as plt
from detecta import detect_peaks
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy.stats.distributions import chi2, norm
from scipy.interpolate import UnivariateSpline, make_smoothing_spline
from scipy.optimize import fminbound
from csaps import csaps
from sklearn.metrics import r2_score




# Class
class OptimalThreshold():
    def __init__(self, data):
        self.data = data

        # Alocate memory space
        self.pks_unicos_valid = None
        self.excedencias_mean_valid = None
        self.excedencias_weight_valid = None
        self.pks = None
        self.locs = None
        self.autocorrelations = None
        self.threshold = None
        self.beta = None
        self.fobj = None
        self.r = None

    def threshold_peak_extraction(self, threshold, n0, min_peak_distance, siglevel=0.05):
        """
        This function identifies peaks in a dataset that exceed a specified 
        threshold and computes statistics such as mean exceedances, variances, 
        and weights for valid unique peaks. Peaks are considered independent if 
        they are separated by a minimum distance.

        Args:
            data                : Input time series or data array
            threshold           : Threshold above which peaks are extracted
            n0                  : Minimum number of exceedances required for valid computation
            min_peak_distance   : Minimum distance between two peaks (in data points)

        Outputs:
            pks_unicos_valid           : Valid unique peaks after removing NaN values
            excedencias_mean_valid     : Mean exceedances for valid peaks
            excedencias_weight_valid   : Weights based on exceedance variance for valid peaks
            pks                        : All detected peaks
            locs                       : Indices of the detected peaks in the data
            autocorrelations           : Lags, correlations and pvalues to check the independence assumption
        """

        # Find peaks exceeding the threshold with specified min distance 
        adjusted_data = np.maximum(self.data - threshold, 0)

        # Usamos la librería detecta que tiene el mismo funcionamiento que la función de matlab findpeaks
        locs = detect_peaks(adjusted_data, mpd=min_peak_distance)
        # Con scipy
        # locs, _ = find_peaks(adjusted_data, distance=min_peak_distance)

        pks = self.data[locs]

        # Calculate autocorrelation for lags 1 to 5 (if enough peaks) 
        num_lags = 5
        if len(pks) > num_lags:
            autocorrelations = np.zeros((num_lags, 3), dtype=float)
            for i in range(num_lags):
                lag = i + 1  
                r, p_value = pearsonr(pks[:-lag], pks[lag:])    # Test corr != 0
                autocorrelations[i, 0] = int(lag)
                autocorrelations[i, 1] = r
                autocorrelations[i, 2] = p_value

                if p_value < siglevel:
                    Warning(f"Lag {int(lag)} significant, consider increase the number of min_peak_distance")
        else:
            # Not enough peaks for autocorrelation analysis
            autocorrelations = np.array([])

        # Unique peaks (pks_unicos), ignoring duplicates 
        pks_unicos = np.unique(pks)

        # Allocate arrays to store mean exceedances, variances, and weights 
        excedencias_mean = np.zeros(len(pks_unicos), dtype=float)
        excedencias_var = np.zeros(len(pks_unicos), dtype=float)
        excedencias_weight = np.zeros(len(pks_unicos), dtype=float)

        # Loop through each unique peak and calculate mean exceedances, variances, and weights 
        for i in range(len(pks_unicos)):
            # Define the current unique peak
            pico_actual = pks_unicos[i]

            # Calculate the exceedances for peaks greater than the current unique peak
            excedencias = pks[pks > pico_actual]

            # If there are enough exceedances (greater than or equal to n0)
            if len(excedencias) >= n0:
                # Compute the mean exceedance (adjusted by the current peak)
                excedencias_mean[i] = np.mean(excedencias) - pico_actual
                # Compute the variance of the exceedances (ddof=1 to use the same variance as matlab)
                excedencias_var[i] = np.var(excedencias, ddof=1)
                # Compute the weight as the number of exceedances divided by the variance
                # Weight = number of exceedances / variance
                # (Guard against division by zero)
                if excedencias_var[i] != 0:
                    excedencias_weight[i] = len(excedencias) / excedencias_var[i]
                else:
                    excedencias_weight[i] = np.nan
            else:
                # If fewer than n0 exceedances, truncate arrays and stop
                excedencias_mean = excedencias_mean[:i]
                excedencias_var = excedencias_var[:i]
                excedencias_weight = excedencias_weight[:i]
                break

        # Trim the list of unique peaks to match the number of valid exceedances
        pks_unicos = pks_unicos[:len(excedencias_weight)]

        # Remove any NaN values from the peak and exceedance data to avoid issues in regression
        valid_indices = (
            ~np.isnan(pks_unicos) &
            ~np.isnan(excedencias_mean) &
            ~np.isnan(excedencias_weight)
        )
        pks_unicos_valid = pks_unicos[valid_indices]
        excedencias_mean_valid = excedencias_mean[valid_indices]
        excedencias_weight_valid = excedencias_weight[valid_indices]


        self.pks_unicos_valid = pks_unicos_valid
        self.excedencias_mean_valid = excedencias_mean_valid
        self.excedencias_weight_valid = excedencias_weight_valid
        self.pks = pks
        self.locs = locs
        self.autocorrelations = autocorrelations
        
        #return pks_unicos_valid, excedencias_mean_valid, excedencias_weight_valid, pks, locs, autocorrelations


    def threshold_studentized_residuals(self, siglevel=0.05, plot_flag=False, filename=None, display_flag=False):
        """
        threshold_studentized_residuals computes the optimal threshold based on Chi-squared
        and studentized residuals. Optionally plots the results if plot_flag is true and 
        displays messages if display_flag is true.
        
        Inputs:
            pks_unicos_valid:           vector of unique peaks (potential thresholds)
            excedencias_mean_valid:     vector of exceedance means
            excedencias_weight_valid:   vector of exceedance weights
            siglevel (optional):        significance level for Chi-squared test (default 0.05)
            plot_flag (optional):       boolean flag, true to plot the graphs, false otherwise
            filename (optional):        path and name for making graphs
            display_flag (optional):    boolean flag, true to display messages, false otherwise
        
        Output:
            threshold:   the optimal threshold found
            beta:        optimal regression coefficients
            fobj:        optimal objective function (weighted leats squares)
            r:           optimal residuals
        """

        stop_search = 0
        it = 1
        threshold = self.pks_unicos_valid[0]     # Initial threshold

        while stop_search == 0 and it <= 10:

            # Find the current threshold in the pks_unicos_valid array
            pos = np.argwhere(self.pks_unicos_valid == threshold).item()
            u_values = self.pks_unicos_valid[pos:]                   # Threshold starting from the current one
            e_values = self.excedencias_mean_valid[pos:]             # Exceedances
            w_values = self.excedencias_weight_valid[pos:]           # Weights

            # Perform the RWLS fitting and calculate studentidez residuals
            beta, fobj, r, rN = RWLSfit(u_values, e_values, w_values)

            # Plot resudals if required
            if plot_flag:
                fig = plt.figure(figsize=(10,6))
                ax = fig.add_subplot(111)
                ax.plot(u_values, rN, "k", linewidth=1.5, label=f"Internally Studentized Residuals\nMin threshold = {threshold.item()}")
                ax.set_xlabel(r"Threshold $u$")
                ax.set_ylabel(r"$r$")
                ax.set_title(f"Studentized Residuals Iteration {it}")
                # ax.text(threshold + 0.5, min(rN) + 0.1 * (max(rN) - min(rN)), f'Min threshold = {threshold}')
                ax.grid()
                ax.legend(loc='upper right')
                if filename is not None:
                    plt.savefig(f"{filename}_StudenRes{it}.png", dpi=300)
                plt.close()

            if fobj > chi2.ppf(1-siglevel, df=u_values.size-2) or np.abs(rN[0]) > norm.ppf(1-siglevel/2,0,1):
                if display_flag:
                    if fobj > chi2.ppf(1-siglevel, df=u_values.size-2):
                        print("Chi-squared test detects anomalies")
                    if np.abs(rN[0]) > norm.ppf(1-siglevel/2,0,1):
                        print("The maximum studentized residual of the first record detects anomalies")
                
                thresholdsearch = 1
            
            else:
                thresholdsearch = 0
                stop_search = 1     # If criteria met, stop the loop

            if thresholdsearch:
                if display_flag:
                    print(f"Maximum sensitivity = {np.max(np.abs(rN))} and thus the optimal threshold seems to be on the right side of the minimum sample value, looking for the location")

                _, threshold = threshold_search(u_values, rN, w_values, plot_flag, f"{filename}_thresholdlocation{it}")
                if display_flag:
                    print(f"New threshold found: {threshold}")

            it += 1

        self.threshold = threshold
        self.beta = beta
        self.fobj = fobj
        self.r = r

        return threshold#,beta,fobj,r

# @staticmethod
def threshold_search(u_data, e_data, W_data, ploteat=False, filename=None):
    """
    Threshold search

    Inputs:
        u_data:     threshold values
        e_data:     exceedances
        W_data:     weights
        ploteat:    flag for plotting
        filename:   file name to save plots

    Outputs:
        fitresult:  fit object representing the fit
        threshold:  the threshold value determined from the fit
    """

    if W_data is None:
        W_data = np.ones(u_data.size)
    if filename is None:
        filename = ""

    orden = np.argsort(u_data)
    u_data = u_data[orden]
    e_data = e_data[orden]
    W_data = W_data[orden]

    # Fit: Smoothing spline
    u_mean = np.mean(u_data)
    u_std = np.std(u_data, ddof=1)
    def objective_function(x):
        return (smoothingspline(u_data, e_data, W_data, u_mean, u_std, x)[0] - 0.9)** 2
    SmoothingParam = fminbound(objective_function, 0.5, 0.99)
    _, fitresult, _ = smoothingspline(u_data, e_data, W_data, u_mean, u_std, SmoothingParam)

    # Find the first zero from the left

    uc = np.linspace(u_data[0], u_data[-1], 2*len(u_data))
    # uc = np.linspace(u_data[0], u_data[-1], 1000)
    ec = fitresult((uc-u_mean)/u_std)
    currentsign = np.sign(ec[0])

    ## If we want to show the fitted smoothing spline
    # if ploteat:
        # plt.figure(figsize=(10,6))
        # plt.plot(u_data,e_data, label="Data")
        # plt.plot(uc, ec, label="Fitted")
        # plt.title("Smoothing Spline Plot")
        # plt.xlabel("Threshold Values (u)")
        # plt.ylabel("Excedaances (e)")
        # plt.grid()
        # plt.show()

    zeroloc = [0, 0] 
    cont = 0
    for i in range(1, len(ec)):
        if currentsign != np.sign(ec[i]):
            # Place midpoint into zeroloc[cont]
            zeroloc[cont] = (uc[i] + uc[i-1]) / 2
            cont += 1
            currentsign = -currentsign
            if cont == 2:
                break

    pos1 = np.argwhere((u_data >= zeroloc[0]) & (u_data <= zeroloc[1]))
    posi = np.argmax(np.abs(e_data[pos1]))
    posi = pos1[0] + posi 
    threshold = u_data[posi]
    mini = e_data[posi]

    if ploteat:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.plot(u_data, e_data, ".k", markersize=1.5, label="Data")
        ax.plot([threshold] * 100, np.linspace(min(e_data), max(e_data), 100), '--', color=[0.5, 0.5, 0.5], linewidth=1.5)
        ax.plot(threshold, mini, "ok", markersize=5, markerfacecolor='w', linewidth=2, label=f"Local optimum = {threshold.item()}")
        ax.set_xlabel(r'Threshold $u$')
        ax.set_ylabel(r'$r^N$')
        ax.legend(loc='upper right')
        ax.grid()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(f"{filename}.png", dpi=300)
        plt.close()
    
    return fitresult, threshold

    # @staticmethod
def smoothingspline(x_data, y_data, w_data, x_mean, x_std, SmoothingParam):
    """
    Fits a smoothing spline to weighted data and calculates the goodness-of-fit (R^2).
    
    Parameters:
    x_data : array-like
        Independent variable.
    y_data : array-like
        Dependent variable.
    w_data : array-like
        Weights for the fit.
    SmoothingParam : float
        Smoothing parameter (controls the tradeoff between smoothness and fit).
    
    Returns:
    qualityparam : float
        R-squared value of the fit.
    fitresult : UnivariateSpline
        The fitted spline model.
    gof : dict
        Goodness-of-fit metrics containing R-squared.
    """    
    # Normalize data
    x_norm = (x_data - x_mean) / x_std

    # Ensure strict increase in x for csaps by deduplicating normalized x
    x_unique, idx = np.unique(x_norm, return_index=True)
    y_use  = y_data[idx]
    w_use  = w_data[idx] 

    # Final safety: if any nonpositive step remains (shouldn't after unique), nudge by eps
    dx = np.diff(x_unique)
    if np.any(dx <= 0):
        eps = np.finfo(float).eps
        bumps = np.maximum.accumulate((dx <= 0).astype(float))
        x_unique = x_unique + np.concatenate([[0.0], bumps]) * eps

    # Usando paquete CSAPS
    spline = csaps(x_unique, y_use, smooth=SmoothingParam, weights=w_use)
    
    # Usando paquete de SCIPY
    # Fit smoothing spline (smoothing parameter scaled by data length)
    # s_value = SmoothingParam * len(x_data)
    # spline = UnivariateSpline(x_norm, y_data, w=w_data, s=s_value)

    # Compute fitted values
    y_fit = spline(x_norm)

    # Compute R-squared
    r2 = r2_score(y_data, y_fit)

    # Store goodness-of-fit metrics
    gof = {'rsquare': r2}

    return r2, spline, gof

    # @staticmethod
def RWLSfit(u, e, w):
    """
    Robust Weighted Least Squares (RWLS) regression.
    
    Parameters:
    u : array-like
        Independent variable (predictor).
    e : array-like
        Dependent variable (response).
    w : array-like
        Weights for the weighted least squares fit.
    
    Returns:
    beta : numpy array
        Estimated regression coefficients [intercept, slope].
    fobj : float
        Objective function value (weighted residual sum of squares).
    r : numpy array
        Residuals.
    rN : numpy array
        Internally studentized residuals.
    """
    if len(u) != len(e) or len(u) != len(w):
        raise ValueError("Error in the number of parameters in function RWLSfit: input arrays must have the same length.")

    # Data size
    n = len(u)

    # Design matrix X with intercept term
    X = np.column_stack((np.ones(n), u))
    Y = np.array(e)

    # Convert weights to diagonal matrix
    W = np.diag(w, 0)  

    # Compute optimal estimates (beta)
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ Y)  # Equivalent to MATLAB: (X'*W*X)\(X'*W*Y)

    # Compute residuals
    r = Y - X @ beta

    # Objective function value (weighted residual sum of squares)
    fobj = r.T @ W @ r

    # Standard deviation of residuals
    sigres = np.sqrt(fobj / (n - 2))

    # Residual variance-covariance matrix
    # Hat or projection matrix
    P = X @ np.linalg.inv(X.T @ W @ X) @ X.T
    # Sensitivity matrix S = I - P * W
    S = np.eye(n) - P @ W

    # Internally studentized residual
    rN = (np.sqrt(np.diag(W)) * r) / (sigres * np.sqrt(1 - np.diag(W) * np.diag(P)))

    return beta, fobj, r, rN






# def threshold_peak_extraction(data, threshold, n0, min_peak_distance):
#     """
#     This function identifies peaks in a dataset that exceed a specified 
#     threshold and computes statistics such as mean exceedances, variances, 
#     and weights for valid unique peaks. Peaks are considered independent if 
#     they are separated by a minimum distance.

#     Args:
#         data                : Input time series or data array
#         threshold           : Threshold above which peaks are extracted
#         n0                  : Minimum number of exceedances required for valid computation
#         min_peak_distance   : Minimum distance between two peaks (in data points)

#     Outputs:
#         pks_unicos_valid           : Valid unique peaks after removing NaN values
#         excedencias_mean_valid     : Mean exceedances for valid peaks
#         excedencias_weight_valid   : Weights based on exceedance variance for valid peaks
#         pks                        : All detected peaks
#         locs                       : Indices of the detected peaks in the data
#         autocorrelations           : Lags, correlations and pvalues to check the independence assumption
#     """

#     # Find peaks exceeding the threshold with specified min distance 
#     adjusted_data = np.maximum(data - threshold, 0)

#     # Usamos la librería detecta que tiene el mismo funcionamiento que la función de matlab findpeaks
#     locs = detect_peaks(adjusted_data, mpd=min_peak_distance)
#     # Con scipy
#     # locs, _ = find_peaks(adjusted_data, distance=min_peak_distance)

#     pks = data[locs]

#     # Calculate autocorrelation for lags 1 to 5 (if enough peaks) 
#     num_lags = 5
#     if len(pks) > num_lags:
#         autocorrelations = np.zeros((num_lags, 3), dtype=float)
#         for i in range(num_lags):
#             lag = i + 1  
#             r, p_value = pearsonr(pks[:-lag], pks[lag:])
#             autocorrelations[i, 0] = lag
#             autocorrelations[i, 1] = r
#             autocorrelations[i, 2] = p_value
#     else:
#         # Not enough peaks for autocorrelation analysis
#         autocorrelations = np.array([])

#     # Unique peaks (pks_unicos), ignoring duplicates 
#     pks_unicos = np.unique(pks)

#     # Allocate arrays to store mean exceedances, variances, and weights 
#     excedencias_mean = np.zeros(len(pks_unicos), dtype=float)
#     excedencias_var = np.zeros(len(pks_unicos), dtype=float)
#     excedencias_weight = np.zeros(len(pks_unicos), dtype=float)

#     # Loop through each unique peak and calculate mean exceedances, variances, and weights 
#     for i in range(len(pks_unicos)):
#         # Define the current unique peak
#         pico_actual = pks_unicos[i]

#         # Calculate the exceedances for peaks greater than the current unique peak
#         excedencias = pks[pks > pico_actual]

#         # If there are enough exceedances (greater than or equal to n0)
#         if len(excedencias) >= n0:
#             # Compute the mean exceedance (adjusted by the current peak)
#             excedencias_mean[i] = np.mean(excedencias) - pico_actual
#             # Compute the variance of the exceedances (ddof=1 to use the same variance as matlab)
#             excedencias_var[i] = np.var(excedencias, ddof=1)
#             # Compute the weight as the number of exceedances divided by the variance
#             # Weight = number of exceedances / variance
#             # (Guard against division by zero)
#             if excedencias_var[i] != 0:
#                 excedencias_weight[i] = len(excedencias) / excedencias_var[i]
#             else:
#                 excedencias_weight[i] = np.nan
#         else:
#             # If fewer than n0 exceedances, truncate arrays and stop
#             excedencias_mean = excedencias_mean[:i]
#             excedencias_var = excedencias_var[:i]
#             excedencias_weight = excedencias_weight[:i]
#             break

#     # Trim the list of unique peaks to match the number of valid exceedances
#     pks_unicos = pks_unicos[:len(excedencias_weight)]

#     # Remove any NaN values from the peak and exceedance data to avoid issues in regression
#     valid_indices = (
#         ~np.isnan(pks_unicos) &
#         ~np.isnan(excedencias_mean) &
#         ~np.isnan(excedencias_weight)
#     )
#     pks_unicos_valid = pks_unicos[valid_indices]
#     excedencias_mean_valid = excedencias_mean[valid_indices]
#     excedencias_weight_valid = excedencias_weight[valid_indices]

    
#     return pks_unicos_valid, excedencias_mean_valid, excedencias_weight_valid, pks, locs, autocorrelations


# def threshold_studentized_residuals(pks_unicos_valid, excedencias_mean_valid, excedencias_weight_valid, siglevel=None, plot_flag=False, filename=None, display_flag=False):
#     """
#     threshold_studentized_residuals computes the optimal threshold based on Chi-squared
#     and studentized residuals. Optionally plots the results if plot_flag is true and 
#     displays messages if display_flag is true.
    
#     Inputs:
#         pks_unicos_valid:           vector of unique peaks (potential thresholds)
#         excedencias_mean_valid:     vector of exceedance means
#         excedencias_weight_valid:   vector of exceedance weights
#         siglevel (optional):        significance level for Chi-squared test (default 0.05)
#         plot_flag (optional):       boolean flag, true to plot the graphs, false otherwise
#         filename (optional):        path and name for making graphs
#         display_flag (optional):    boolean flag, true to display messages, false otherwise
    
#     Output:
#         threshold:   the optimal threshold found
#         beta:        optimal regression coefficients
#         fobj:        optimal objective function (weighted leats squares)
#         r:           optimal residuals
#     """

#     if siglevel is None:
#         siglevel = 0.05


#     stop_search = 0
#     it = 1
#     threshold = pks_unicos_valid[0]     # Initial threshold

#     while stop_search == 0 and it <= 10:

#         # Find the current threshold in the pks_unicos_valid array
#         pos = np.argwhere(pks_unicos_valid == threshold).item()
#         u_values = pks_unicos_valid[pos:]                   # Threshold starting from the current one
#         e_values = excedencias_mean_valid[pos:]             # Exceedances
#         w_values = excedencias_weight_valid[pos:]           # Weights

#         # Perform the RWLS fitting and calculate studentidez residuals
#         beta, fobj, r, rN = RWLSfit(u_values, e_values, w_values)

#         # Plot resudals if required
#         if plot_flag:
#             fig = plt.figure(figsize=(10,6))
#             ax = fig.add_subplot(111)
#             ax.plot(u_values, rN, "k", linewidth=1.5, label=f"Internally Studentized Residuals\nMin threshold = {threshold.item()}")
#             ax.set_xlabel(r"Threshold $u$")
#             ax.set_ylabel(r"$r$")
#             ax.set_title(f"Studentized Residuals Iteration {it}")
#             # ax.text(threshold + 0.5, min(rN) + 0.1 * (max(rN) - min(rN)), f'Min threshold = {threshold}')
#             ax.grid()
#             ax.legend(loc='upper right')
#             if filename is not None:
#                 plt.savefig(f"{filename}_StudenRes{it}.png", dpi=300)
#             plt.show()

#         if fobj > chi2.ppf(1-siglevel, df=u_values.size-2) or np.abs(rN[0]) > norm.ppf(1-siglevel/2,0,1):
#             if display_flag:
#                 if fobj > chi2.ppf(1-siglevel, df=u_values.size-2):
#                     print("Chi-squared test detects anomalies")
#                 if np.abs(rN[0]) > norm.ppf(1-siglevel/2,0,1):
#                     print("The maximum studentized residual of the first record detects anomalies")
            
#             thresholdsearch = 1
        
#         else:
#             thresholdsearch = 0
#             stop_search = 1     # If criteria met, stop the loop

#         if thresholdsearch:
#             if display_flag:
#                 print(f"Maximum sensitivity = {np.max(np.abs(rN))} and thus the optimal threshold seems to be on the right side of the minimum sample value, looking for the location")

#             _, threshold = threshold_search(u_values, rN, w_values, plot_flag, f"{filename}_thresholdlocation{it}")
#             if display_flag:
#                 print(f"New threshold found: {threshold}")

#         it += 1

#     return threshold,beta,fobj,r

# def threshold_search(u_data, e_data, W_data, ploteat=False, filename=None):
#     """
#     Threshold search

#     Inputs:
#         u_data:     threshold values
#         e_data:     exceedances
#         W_data:     weights
#         ploteat:    flag for plotting
#         filename:   file name to save plots

#     Outputs:
#         fitresult:  fit object representing the fit
#         threshold:  the threshold value determined from the fit
#     """

#     if W_data is None:
#         W_data = np.ones(u_data.size)
#     if filename is None:
#         filename = ""

#     orden = np.argsort(u_data)
#     u_data = u_data[orden]
#     e_data = e_data[orden]
#     W_data = W_data[orden]

#     # Fit: Smoothing spline
#     u_mean = np.mean(u_data)
#     u_std = np.std(u_data, ddof=1)
#     objective_function = lambda x: (smoothingspline(u_data, e_data, W_data, u_mean, u_std, x)[0] - 0.9)** 2
#     SmoothingParam = fminbound(objective_function, 0.5, 0.99)
#     _, fitresult, _ = smoothingspline(u_data, e_data, W_data, u_mean, u_std, SmoothingParam)

#     # Find the first zero from the left
#     uc = np.linspace(u_data[0], u_data[-1], 1000)
#     ec = fitresult((uc-u_mean)/u_std)
#     currentsign = np.sign(ec[0])

#     ## If we want to show the fitted smoothing spline
#     # if ploteat:
#         # plt.figure(figsize=(10,6))
#         # plt.plot(u_data,e_data, label="Data")
#         # plt.plot(uc, ec, label="Fitted")
#         # plt.title("Smoothing Spline Plot")
#         # plt.xlabel("Threshold Values (u)")
#         # plt.ylabel("Excedaances (e)")
#         # plt.grid()
#         # plt.show()

#     zeroloc = [0, 0] 
#     cont = 0
#     for i in range(1, len(ec)):
#         if currentsign != np.sign(ec[i]):
#             # Place midpoint into zeroloc[cont]
#             zeroloc[cont] = (uc[i] + uc[i-1]) / 2
#             cont += 1
#             currentsign = -currentsign
#             if cont == 2:
#                 break

#     pos1 = np.argwhere((u_data >= zeroloc[0]) & (u_data <= zeroloc[1]))
#     posi = np.argmax(np.abs(e_data[pos1]))
#     posi = pos1[0] + posi 
#     threshold = u_data[posi]
#     mini = e_data[posi]

#     if ploteat:
#         fig = plt.figure(figsize=(10,6))
#         ax = fig.add_subplot(111)
#         ax.plot(u_data, e_data, ".k", markersize=1.5, label="Data")
#         ax.plot([threshold] * 100, np.linspace(min(e_data), max(e_data), 100), '--', color=[0.5, 0.5, 0.5], linewidth=1.5)
#         ax.plot(threshold, mini, "ok", markersize=5, markerfacecolor='w', linewidth=2, label=f"Local optimum = {threshold.item()}")
#         ax.set_xlabel(r'Threshold $u$')
#         ax.set_ylabel(r'$r^N$')
#         ax.legend(loc='upper right')
#         ax.grid()
#         plt.xticks(fontsize=14)
#         plt.yticks(fontsize=14)
#         plt.tight_layout()
#         if filename is not None:
#             plt.savefig(f"{filename}.png", dpi=300)
#         plt.show()
    
#     return fitresult, threshold

# def smoothingspline(x_data, y_data, w_data, x_mean, x_std, SmoothingParam):
#     """
#     Fits a smoothing spline to weighted data and calculates the goodness-of-fit (R^2).
    
#     Parameters:
#     x_data : array-like
#         Independent variable.
#     y_data : array-like
#         Dependent variable.
#     w_data : array-like
#         Weights for the fit.
#     SmoothingParam : float
#         Smoothing parameter (controls the tradeoff between smoothness and fit).
    
#     Returns:
#     qualityparam : float
#         R-squared value of the fit.
#     fitresult : UnivariateSpline
#         The fitted spline model.
#     gof : dict
#         Goodness-of-fit metrics containing R-squared.
#     """
#     # Normalize data
#     x_norm = (x_data - x_mean) / x_std

#     # Usando paquete CSAPS
#     spline = csaps(x_norm, y_data, smooth=SmoothingParam, weights=w_data)
    
#     # Usando paquete de SCIPY
#     # Fit smoothing spline (smoothing parameter scaled by data length)
#     # s_value = SmoothingParam * len(x_data)
#     # spline = UnivariateSpline(x_norm, y_data, w=w_data, s=s_value)

#     # Compute fitted values
#     y_fit = spline(x_norm)

#     # Compute R-squared
#     r2 = r2_score(y_data, y_fit)

#     # Store goodness-of-fit metrics
#     gof = {'rsquare': r2}

#     return r2, spline, gof


# def RWLSfit(u, e, w):
#     """
#     Robust Weighted Least Squares (RWLS) regression.
    
#     Parameters:
#     u : array-like
#         Independent variable (predictor).
#     e : array-like
#         Dependent variable (response).
#     w : array-like
#         Weights for the weighted least squares fit.
    
#     Returns:
#     beta : numpy array
#         Estimated regression coefficients [intercept, slope].
#     fobj : float
#         Objective function value (weighted residual sum of squares).
#     r : numpy array
#         Residuals.
#     rN : numpy array
#         Internally studentized residuals.
#     """
#     if len(u) != len(e) or len(u) != len(w):
#         raise ValueError("Error in the number of parameters in function RWLSfit: input arrays must have the same length.")

#     # Data size
#     n = len(u)

#     # Design matrix X with intercept term
#     X = np.column_stack((np.ones(n), u))
#     Y = np.array(e)

#     # Convert weights to diagonal matrix
#     W = np.diag(w, 0)  

#     # Compute optimal estimates (beta)
#     beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ Y)  # Equivalent to MATLAB: (X'*W*X)\(X'*W*Y)

#     # Compute residuals
#     r = Y - X @ beta

#     # Objective function value (weighted residual sum of squares)
#     fobj = r.T @ W @ r

#     # Standard deviation of residuals
#     sigres = np.sqrt(fobj / (n - 2))

#     # Residual variance-covariance matrix
#     # Hat or projection matrix
#     P = X @ np.linalg.inv(X.T @ W @ X) @ X.T
#     # Sensitivity matrix S = I - P * W
#     S = np.eye(n) - P @ W

#     # Internally studentized residual
#     rN = (np.sqrt(np.diag(W)) * r) / (sigres * np.sqrt(1 - np.diag(W) * np.diag(P)))

#     return beta, fobj, r, rN

