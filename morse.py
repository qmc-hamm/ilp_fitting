import numpy as np
import pandas as pd
import scipy.optimize

def morse(r, D, r0, a, E0):
    return D*(1 - np.exp(-a*(r - r0)))**2 + E0

def morse_fit(x, y, x_range=1):
    '''
    Performs a morse fit from given `x` and `y`.
    Returns a list of the fitting parameters: D, r0, a, E0
    '''
    # find a starting guess value `D` for the morse fit by doing a quadratic fit near the minimum
    d_eq_idx = y.idxmin()
    d_eq = x[d_eq_idx]
    mask = (d_eq - x_range < x) & (x < d_eq + x_range)
    xx = x[mask]
    yy = y[mask]
    fit = np.polyfit(xx, yy, 2)

    # the starting guess for D is D0 = fit[0]
    # this is because at near minimum r = r0 + xi, and xi << 1, y(xi) = D*a**2*xi**2 + E0
    p0 = [fit[0], 3.3, 1.0, np.min(y)]
    popt, pcov = scipy.optimize.curve_fit(morse, x, y, p0)
    return popt

def get_e_inf(g):
    popt = morse_fit(g['d'], g['energy'])
    e_inf = popt[0] + popt[-1]
    return pd.DataFrame({'e_inf': [e_inf]})
