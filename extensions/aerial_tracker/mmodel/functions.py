import numpy as np
#from .utils import cubic_spline_natural_1d, cubic_spline_notAknot_1d#, cubic_spline_clamped_1d
from scipy.optimize import curve_fit
import time
import copy

def func_linear(x, a, b):
    return a * x + b

def func_poly2(x, a, b, c):
    return a * (x**2) + b * x + c

def func_poly3(x, a, b, c, d):
    return a*(x**3) + b*(x**2) + c*x + d

def concat_cols(x, y):
    return np.hstack([x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)])

def get_fitted_path(f, t, x):
    popt_x, pcov_x = curve_fit(f, t, x)
    tr_x = np.trace(pcov_x)/pcov_x.shape[0] 
    fitted_x = f(t, *popt_x)
    return fitted_x, tr_x

def optimal_fit(t, x):
    best_tr = 1e9
    best_path = None
    fs = [func_linear, func_poly2, func_poly3]
    # NOTE: for now we only extrapolate object's path based on linear model. Higher degree
    # polynomials are too sensitive facing outlier. The best case is using them when added
    # outlier removal facility before fitting
    #fs = [func_linear]
    for f in fs:
        path, tr = get_fitted_path(f, t, x) 
        if tr < best_tr:
            best_tr = tr
            best_path = copy.deepcopy(path)
    return best_path


# def cubic_spline_bcNotAknot(ts, points, new_times=None, t=None):
#     return np.vstack((cubic_spline_notAknot_1d(ts, points[:,0], new_times, t),
#                       cubic_spline_notAknot_1d(ts, points[:,1], new_times, t))).T

# def cubic_spline_bcClamped(ts, points, new_times=None, t=None):
#     return np.vstack((cubic_spline_clamped_1d(ts, points[:,0], new_times, t),
#                       cubic_spline_clamped_1d(ts, points[:,1], new_times, t))).T

def cubic_spline_bcNatural(ts, points, new_times=None, t=None):
    return np.vstack((cubic_spline_natural_1d(ts, points[:,0], new_times, t),
                      cubic_spline_natural_1d(ts, points[:,1], new_times, t))).T

def linear_delayed(ts, _pos_buff, new_times=None, t=None, delay=2):#,_interp_factor=1):
    ## calculate velocities for each consecutive pair of buffered target positions (used to 
    ## estimate a current velocity for the target)
    v = (_pos_buff[-1-delay] - _pos_buff[-2-delay]) / (ts[-1-delay] - ts[-2-delay])
    if not new_times is None:
        outs = []
        for n in new_times:
            dt = n - ts[-1]
            outs.append(_pos_buff[-1] + v*dt)
            # outs.append(_pos_buff[-1] + _interp_factor*v*dt)
        return outs  
    elif not t is None:
        dt = t - ts[-1]
        out = _pos_buff[-1] + v*dt
        # out = _pos_buff[-1] + _interp_factor*v*dt
        return out
    else:
        raise Exception("One of 'new_times' or 't' must be set")


def linear_curve_fit(ts, _pos_buff, new_times=None, t=None):#,_interp_factor=1):
    fitted_path = optimal_fit(ts, _pos_buff)
    return linear_avg_velocity(ts, fitted_path, new_times, t)


def linear_avg_velocity(ts, _pos_buff, new_times=None, t=None):#,_interp_factor=1):
    ## calculate velocities for each consecutive pair of buffered target positions (used to 
    ## estimate a current velocity for the target)
    vs = []
    for i in range(1, _pos_buff.shape[1]):
        t0 = ts[i-1]
        pos0 = _pos_buff[i-1]
        t = ts[i]
        pos = _pos_buff[i]
        dx = np.array(pos) - np.array(pos0)
        dt = t-t0
        if dt != 0:
            vs.append(dx/dt)
    if len(vs)==0:
        return None
    v = np.mean(vs,0)
    if not new_times is None:
        outs = []
        for n in new_times:
            dt = n - ts[-1]
            outs.append(_pos_buff[-1] + v*dt)
            # outs.append(_pos_buff[-1] + _interp_factor*v*dt)
        return outs  
    elif not t is None:
        dt = t - ts[-1]
        out = _pos_buff[-1] + v*dt
        # out = _pos_buff[-1] + _interp_factor*v*dt
        return out
    else:
        raise Exception("One of 'new_times' or 't' must be set")

