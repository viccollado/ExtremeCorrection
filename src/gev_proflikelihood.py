import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import minimize, root_scalar
from scipy.interpolate import interp1d

from .gev_utils import q_gev

def gev_rp_plik(params, # Estimated parameters of GEV [location, scale, shape]
                nllopt, # Optimal negative log-likelihood value for estimated parameters
                data, # Data used for estimation
                year,  # Year to compute return level
                xlow, # Bound of return level
                xup, # Bound of return level
                conf=0.95, 
                nint=100,
                plot=False):
    """
    Function to compute the confidence interval of the return level using profile likelihood method

    Parameters
    ----------
    params : list
        Estimated parameters of GEV [location, scale, shape]
    nllopt : float
        Optimal negative log-likelihood value for estimated parameters
    data : np.array
        Data used for estimation
    year : int
        Year to compute return level
    xlow : float
        Lower bound of return level
    xup : float
        Upper bound of return level
    conf : float, optional
        Confidence level, by default 0.95
    nint : int, optional
        Number of points to evaluate profile likelihood, by default 100
    plot : bool, optional
        Whether to plot the profile likelihood, by default False
    
    Returns
    -------
    conf_int : list
        Confidence interval of the return level
    
    Example
    -------
    >>> from scipy.stats import genextreme
    >>> np.random.seed(42)
    >>> data = genextreme.rvs(-0.1, loc=0, scale=1, size=1000)
    >>> fitted_params = genextreme.fit(data=data)
    >>> initial_params0 = [fitted_params[1], fitted_params[2], fitted_params[0]]  # [location, scale, shape]
    >>> initial_params1 = [fitted_params[1], fitted_params[2], -fitted_params[0]]  # [location, scale, shape]
    >>> print(f"Fitted GEV parameters {initial_params1}")

    >>> from gev_utils import nll_gev, q_gev
    >>> # initial_params = [0, 1, -0.1]  # Initial guess for [location, scale, -shape]
    >>> nllopt = nll_gev(data, initial_params0)
     
    >>> conf_int = gev_rp_plik(initial_params1, nllopt, data, year=100, xlow=0, xup=10, plot=True, nint=1000)
    >>> print(f"100-years Return level {q_gev(1-1/100, initial_params0)}")
    >>> print("95% Confidence interval for 100-year return level:", conf_int)

    """    
    if year <= 1:
        raise ValueError("`year' must be greater than one")

    
    probs = 1/year
    v = np.zeros(nint)
    # opt_params_plik = np.zeros((nint, 2))
    x = np.linspace(xlow, xup, nint)

    # Initial guess for scale and shape
    initial_guess = [params[1], params[2]]

    # Use location to estimate the return level conf intervals

    for i, zp in enumerate(x):
        # Profile log-likelihood of GEV for fixed return level zp
        def gev_plik(a):
            # Computes profile neg log likelihood
            if np.abs(a[1]) < 1e-6: # If shape is close to zero use Gumbel
                mu = zp + a[0] * np.log(-np.log(1 - probs))
                y = (data - mu) / a[0]

                if np.isinf(mu) or a[0] <= 0:
                    nll = 1e6
                else:
                    nll = len(y) * np.log(a[0]) + np.sum(np.exp(-y)) + np.sum(y)
            else:
                mu = zp - a[0] / a[1] * ((- np.log(1 - probs)) ** (-a[1])-1)
                y = (data - mu) / a[0]
                y = 1 + a[1] * y 
                if np.isinf(mu) or a[0] <= 0 or np.any(y <= 0):
                    nll = 1e6
                else:
                    nll = len(y) * np.log(a[0]) + np.sum(y**(-1/a[1])) + np.sum(np.log(y)) * (1/a[1] + 1)

            return nll
        
        opt = minimize(gev_plik, initial_guess, method="Nelder-Mead")
        initial_guess = opt.x
        # opt_params_plik[i, :] = opt.x
        v[i] = opt.fun

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, -v, label="Profile log-likelihood", markersize=0, linewidth=2)
        ax.axhline(y = -nllopt, color="r", linestyle="--", label="Optimal log-likelihood")
        ax.axhline(y = -nllopt - 0.5 * chi2.ppf(conf, 1), color="r")
        ax.set_xlabel("Return level")
        ax.set_ylabel("Profile log-likelihood")
        ax.legend()
        plt.show()

    # Find the roots of the equation: -v = -nllopt - 0.5*chi2.ppf(conf, 1)
    ret_level_base_bound = q_gev(1 - probs, p = [params[0], params[1], -params[2]])
    # Use root_scalar to find the roots
    f_interpolated = interp1d(x, -v)
    def froot_gev_plik(zp):
        return f_interpolated(zp) + nllopt + 0.5 * chi2.ppf(conf, 1)
    
    sol_lower = root_scalar(froot_gev_plik, bracket=[xlow, ret_level_base_bound], method="bisect")
    sol_upper = root_scalar(froot_gev_plik, bracket=[ret_level_base_bound, xup], method="bisect")

    conf_int = [sol_lower.root, sol_upper.root]

    # # OR take the closest indices and then obtain the conf interval using these indices
    # idx_sorted = np.argsort(np.abs(-v - (-nllopt - 0.5 * chi2.ppf(conf, 1))))

    # conf_int = x[idx_sorted[0]], x[idx_sorted[1]]
    # conf_int = np.sort(conf_int)    # Sort the interval

    return conf_int 





# Test the function
if __name__ == "__main__":
    # Example usage
    from scipy.stats import genextreme
    np.random.seed(42)
    data = genextreme.rvs(0.1, loc=0, scale=1, size=1000)
    fitted_params = genextreme.fit(data=data)
    initial_params0 = [fitted_params[1], fitted_params[2], fitted_params[0]]  # [location, scale, shape]
    initial_params1 = [fitted_params[1], fitted_params[2], -fitted_params[0]]  # [location, scale, shape]
    print(f"Fitted GEV parameters {initial_params1}")

    from gev_utils import nll_gev, q_gev
    # initial_params = [0, 1, -0.1]  # Initial guess for [location, scale, -shape]
    nllopt = nll_gev(data, initial_params0)

    conf_int = gev_rp_plik(initial_params1, nllopt, data, year=2, xlow=0, xup=10, plot=True, nint=1000)
    print(f"100-years Return level {q_gev(1-1/2, initial_params0)}")
    print("95% Confidence interval for 100-year return level:", conf_int)

    times = [1.5, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    len_times = len(times)
    zp_real = np.zeros(len_times)
    zp_conf_int_low = np.zeros(len_times)
    zp_conf_int_upp = np.zeros(len_times)
    for i, zp in enumerate(times):
        zp_real[i] = q_gev(1-1/zp, p=initial_params0)
        conf_int = gev_rp_plik(initial_params1, nllopt, data, year=zp, xlow=-1-i, xup=2*(i+1), plot=False, nint=1000)
        zp_conf_int_low[i] = conf_int[0]
        zp_conf_int_upp[i] = conf_int[1]

    plt.figure()
    plt.semilogx(times, zp_real)
    plt.semilogx(times, zp_conf_int_low, "r--")
    plt.semilogx(times, zp_conf_int_upp, "r--")
    plt.show()