import scipy as sp
import numpy as np

def output_model(model, new_times, t):
    if not new_times is None:
        new_vals = []
        for n in new_times:
            new_vals.append(model(n))
        return new_vals
    elif not t is None:
        return model(t)
    else:
        raise Exception("One of 'new_times' or 't' must be set")

def cubic_spline_notAknot_1d(ts, vals, new_times=None, t=None):
    return output_model(sp.interpolate.CubicSpline(ts, vals, bc_type='not-a-knot'), 
                        new_times, t)

def cubic_spline_natural_1d(ts, vals, new_times=None, t=None):
    return output_model(sp.interpolate.CubicSpline(ts, vals, bc_type='natural'), 
                        new_times, t)

def cubic_spline_clamped_1d(ts, vals, new_times=None, t=None):
    return output_model(sp.interpolate.CubicSpline(ts, vals, bc_type='clamped'), 
                        new_times, t)

def linear_avg_velocity_1d(ts, _pos_buff, new_times=None, t=None):#,_interp_factor=1):
    ## calculate velocities for each consecutive pair of buffered target positions (used to 
    ## estimate a current velocity for the target)
    vs = []
    for i in range(1, _pos_buff.shape[0]):
        t0 = ts[i-1]
        pos0 = _pos_buff[i-1]
        t_ = ts[i]
        pos = _pos_buff[i]
        dx = np.array(pos) - np.array(pos0)
        dt = t_-t0
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

