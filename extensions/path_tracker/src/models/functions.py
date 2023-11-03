import numpy as np
from .utils import cubic_spline_natural_1d, quadratic_spline_1d, cubic_spline_keepCurve_1d

def quadratic_spline(ts, points, new_times=None, t=None):
    return np.vstack((quadratic_spline_1d(ts, points[:,0], new_times, t),
                      quadratic_spline_1d(ts, points[:,1], new_times, t))).T

def cubic_spline_keepCurve(ts, points, new_times=None, t=None):
    return np.vstack((cubic_spline_keepCurve_1d(ts, points[:,0], new_times, t),
                      cubic_spline_keepCurve_1d(ts, points[:,1], new_times, t))).T

# def cubic_spline_bcNatural(ts, points, new_times=None, t=None):
#     return np.vstack((cubic_spline_natural_1d(ts, points[:,0], new_times, t),
#                       cubic_spline_natural_1d(ts, points[:,1], new_times, t))).T

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

# def linear(ts, _pos_buff, new_times=None, t=None):#,_interp_factor=1):
#     ## calculate velocities for each consecutive pair of buffered target positions (used to 
#     ## estimate a current velocity for the target)
#     v = (_pos_buff[-1] - _pos_buff[-2]) / (ts[-1] - ts[-2])
#     if not new_times is None:
#         outs = []
#         for n in new_times:
#             dt = n - ts[-1]
#             outs.append(_pos_buff[-1] + v*dt)
#             # outs.append(_pos_buff[-1] + _interp_factor*v*dt)
#         return outs  
#     elif not t is None:
#         dt = t - ts[-1]
#         out = _pos_buff[-1] + v*dt
#         # out = _pos_buff[-1] + _interp_factor*v*dt
#         return out
#     else:
#         raise Exception("One of 'new_times' or 't' must be set")

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

