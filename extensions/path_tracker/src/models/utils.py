import scipy.interpolate as spi
import numpy as np

def calc_1st_dv(inp, t, idx):
    if idx == 0 or idx >= len(t) or len(inp) != len(t):
        raise Exception("bad input")
    return ((inp[idx]-inp[idx-1])/(t[idx]-t[idx-1]))

def calc_2nd_dv(inp, t, idx=None):
    if idx is None:
        idx = len(t)-1
    if idx <= 1 or idx >= len(t) or len(inp) != len(t):
        raise Exception("bad input")
    return ((calc_1st_dv(inp, t, idx)-calc_1st_dv(inp, t, idx))/(t[idx]-t[idx-1]))

def calc_1st_dv_all(inp, t):
    out = []
    for k in range(1, len(inp)):
        out.append((inp[k]-inp[k-1])/(t[k]-t[k-1]))
    return out

def calc_2nd_dv_all(inp, t):
    out = []
    out = []
    for k in range(2, len(inp)):
        out.append((calc_1st_dv(inp, t, k)-calc_1st_dv(inp, t, k-1))/(t[k]-t[k-1]))
    return out

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
    return output_model(spi.CubicSpline(ts, vals, bc_type='not-a-knot'), 
                        new_times, t)

def cubic_spline_natural_1d(ts, vals, new_times=None, t=None):
    return output_model(spi.CubicSpline(ts, vals, bc_type='natural'), 
                        new_times, t)

def cubic_spline_keepCurve_1d(ts, vals, new_times=None, t=None):
    my_end_dd = calc_2nd_dv(vals, ts)
    my_start_dd = calc_2nd_dv(vals, ts, 2)
    model = spi.CubicSpline(ts, vals, bc_type=((2, my_start_dd), (2, my_end_dd)))
    return output_model(model, new_times, t)

def cubic_spline_clamped_1d(ts, vals, new_times=None, t=None):
    return output_model(spi.CubicSpline(ts, vals, bc_type='clamped'), 
                        new_times, t)

def quadratic_spline_1d(ts, vals, new_times=None, t=None):
    return output_model(spi.interp1d(ts, vals, kind='quadratic', 
                                                fill_value='extrapolate'), new_times, t)

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

