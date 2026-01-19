import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import minimize, root_scalar
from scipy.interpolate import interp1d

from .pot_utils import q_pot, nll_pot

def gpdpoiss_rp_plik(params, # Estimated parameters of GEV [threshold, scale, shape, lambda]
                    nllopt, # Optimal negative log-likelihood value for estimated parameters
                    data, # Data used for estimation
                    m,  # Year to compute return level,
                    n_years, # Number of years in the data
                    xlow, # Bound of return level
                    xup, # Bound of return level
                    # npy=365, # Number of observations per year,
                    conf=0.95, 
                    nint=1000,
                    plot=False,
                    save_file=None):
    """
    Function to compute the confidence interval of the return level using profile likelihood method
    for annual maxima using GPD-Poisson model.

    Parameters
    ----------
    params : list
        Estimated parameters of GEV [threshold, scale, shape, lambda]
    nllopt : float
        Optimal negative log-likelihood value for estimated parameters
    data : np.array
        Data used for estimation
    m : int
        Return level (probability 1/m)
    npy : int DEPRECATED USED FOR GPD ONLY
        Number of observations per year
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
    >>> from scipy.stats import genpareto
    >>> np.random.seed(42)

    >>> # Simulate GPD data
    >>> threshold = 0
    >>> shape = 0.1
    >>> scale = 1
    >>> lam = 5  # Poisson rate (mean number of exceedances per year)
    >>> n_years = 20
    >>> n_samples = int(lam * n_years)
    >>> data = genpareto.rvs(c=shape, loc=threshold, scale=scale, size=n_samples)

    >>> # GPD fit
    >>> fit_shape, fit_loc, fit_scale = genpareto.fit(data, floc=threshold)
    >>> params = [threshold, fit_scale, fit_shape, lam]  # [threshold, scale, shape, lambda]
    >>> print(f"Fitted GPD-Poiss parameters {params}")
    
    >>> from pot_utils import nll_pot, q_pot
    >>> nllopt = nll_pot(data, p=[threshold, lam, fit_scale, fit_shape], n_years=n_years)
    >>> conf_int = gpdpoiss_rp_plik(params, nllopt, data, m=10, n_years=n_years, xlow=1, xup=500, plot=True, nint=500, npy=5)
    >>> print(f"100-years Return level {q_pot(1-1/10, p=[threshold, fit_scale, fit_shape], lam=lam)}")
    >>> print("95% Confidence interval for 100-year return level:", conf_int)

    """    
    if m < 1:
        raise ValueError("`year' must be greater than one")

    exceedances = data[data > params[0]]
    # m = m * npy
    probs = 1/m
    v = np.zeros(nint)
    # opt_params_plik = np.zeros((nint, 2))
    x = np.linspace(xlow, xup, nint)

    # Initial guess for scale and shape
    initial_guess = params[2]

    # Use location to estimate the return level conf intervals

    for i, zp in enumerate(x):
        # Profile log-likelihood of GEV for fixed return level zp
        def gpdpoiss_plik(a):
            # a[0] = threshold, a[1] = shape, a[2] = lambda (dejar fijo lambda)
            # a = shape
            # if m != np.inf: 
            #     scale = (a * (zp - params[0]))/((m * params[3])**a - 1)
            # else:
            #     scale = (params[0] - zp)/a

            if np.abs(a) < 1e-6:
                scale = (params[0] - zp)/np.log(-np.log((1-probs)/params[3]))
                if scale <= 0:
                    nll = 1e6
                else:
                    # nll = - len(data)*np.log(params[3]) + n_years*params[3] + len(exceedances) * np.log(scale) + np.sum(exceedances - params[0])/scale
                    nll = nll_pot(data, n_years, [params[0], params[3], scale, a])
                # nll = len(data) * np.log(scale) + np.sum(data - params[0])/scale # For GPD
            else:
                scale = (params[0]-zp)*a/(1-(-np.log(1-probs)/params[3])**(-a))
                if scale <= 0:
                    nll = 1e6
                else:
                    y = (exceedances - params[0])/scale 
                    y = 1 + a * y
                    if any(y <= 0) or scale <= 0:
                        nll = 1e6
                    else:
                        # nll = - len(exceedances)*np.log(params[3]) + n_years*params[3] + len(exceedances)*np.log(scale) + (1+1/a)*np.sum(np.log(y))
                        nll = nll_pot(data, n_years, [params[0], params[3], scale, a])
                        # FOR GPD 
                        # # nll = len(data) * np.log(scale) + np.sum(np.log(y)) * (1/a + 1) # For GPD
            return nll
        
        opt = minimize(gpdpoiss_plik, initial_guess, method="L-BFGS-B")
        # opt = minimize_scalar(gpdpoiss_plik, bracket=[initial_guess-0.2, initial_guess+0.2], method="Brent")
        initial_guess = opt.x
        # opt_params_plik[i, :] = opt.x
        v[i] = opt.fun

    # nllopt = np.min(v)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, -v, label="Profile log-likelihood", markersize=0, linewidth=2)
        ax.axhline(y = -nllopt, color="r", linestyle="--", label="Optimal log-likelihood")
        ax.axhline(y = -nllopt - 0.5 * chi2.ppf(conf, 1), color="r")
        ax.set_xlabel("Return level")
        ax.set_ylabel("Profile log-likelihood")
        ax.legend()
        ax.set_ylim(-nllopt - 2, max(-v) + 1)
        if save_file is not None:
            plt.savefig(f"Figures/Hs_Santona_Tp/ReturnLevels/{save_file}_gpdpoiss_return_level.png", dpi=200)
            plt.close()
        # plt.show()

    # Find the roots of the equation: -v = -nllopt - 0.5*chi2.ppf(conf, 1)
    ret_level_base_bound = q_pot(1 - probs, p = [params[0], params[1], params[2]], lam=params[3])
    # Use root_scalar to find the roots
    f_interpolated = interp1d(x, -v, kind="linear")
    def froot_gpdpoiss_plik(zp):
        return f_interpolated(zp) + nllopt + 0.5 * chi2.ppf(conf, 1)
    
    sol_lower = root_scalar(froot_gpdpoiss_plik, bracket=[xlow, ret_level_base_bound], method="bisect")
    sol_upper = root_scalar(froot_gpdpoiss_plik, bracket=[ret_level_base_bound, xup], method="bisect")

    conf_int = [sol_lower.root, sol_upper.root]

    # # OR take the closest indices and then obtain the conf interval using these indices
    # idx_sorted = np.argsort(np.abs(-v - (-nllopt - 0.5 * chi2.ppf(conf, 1))))

    # conf_int = x[idx_sorted[0]], x[idx_sorted[1]]
    # conf_int = np.sort(conf_int)    # Sort the interval

    return conf_int 

# Test the function
if __name__ == "__main__":
    # Example usage
    from scipy.stats import genpareto
    np.random.seed(42)
    # Simulate GPD data
    threshold = 13.119243621826172
    shape = -0.324300076783393
    scale = 2.2173392137448147
    lam = 37.25  # Poisson rate (mean number of exceedances per year)
    n_years = 25
    n_samples = int(lam * n_years)
    data = genpareto.rvs(c=shape, loc=threshold, scale=scale, size=n_samples)
    # GPD fit
    fit_shape, fit_loc, fit_scale = genpareto.fit(data, floc=threshold)
    params = [threshold, fit_scale, fit_shape, lam]  # [threshold, scale, shape, lambda]
    print(f"Fitted GPD-Poiss parameters {params}")

    from pot_utils import nll_pot, q_pot
    # initial_params = [0, 1, -0.1]  # Initial guess for [location, scale, -shape]
    nllopt = nll_pot(data, p=[threshold, lam, fit_scale, fit_shape], n_years=n_years)

    conf_int = gpdpoiss_rp_plik(params, nllopt, data, m=10, n_years=n_years, xlow=10, xup=40, plot=True, nint=500)
    print(f"100-years Return level {q_pot(1-1/1.001, p=[threshold, fit_scale, fit_shape], lam=lam)}")
    print("95% Confidence interval for 100-year return level:", conf_int)

    zp_real = np.zeros(8)
    zp_conf_int_low = np.zeros(8)
    zp_conf_int_upp = np.zeros(8)
    times = [1.1, 5, 10, 20, 50, 100, 200, 500]
    for i, zp in enumerate(times):
        zp_real[i] = q_pot(1-1/zp, p=[threshold, fit_scale, fit_shape], lam=lam)
        conf_int = gpdpoiss_rp_plik(params, nllopt, data, m=zp, n_years=n_years, xlow=10, xup=40, plot=False, nint=500)
        zp_conf_int_low[i] = conf_int[0]
        zp_conf_int_upp[i] = conf_int[1]

    plt.figure()
    plt.semilogx(times, zp_real)
    plt.semilogx(times, zp_conf_int_low, "r--")
    plt.semilogx(times, zp_conf_int_upp, "r--")
    plt.show()
