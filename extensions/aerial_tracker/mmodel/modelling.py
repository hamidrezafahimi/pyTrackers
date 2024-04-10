from enum import Enum
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()) + "/..")
from .pathBuffering import PathBufferer1D
from .functions import optimal_fit
#from .utils import cubic_spline_natural_1d, linear_avg_velocity_1d
import numpy as np

class ModelType1D(Enum):
    LINEAR_EXTRAP = 1
    LINEAR_FIT = 2
    CUBIC_SPLINE = 3

class Modeller1D:
    def __init__(self, type: ModelType1D):
        self.buffer = PathBufferer1D()
        if type == ModelType1D.CUBIC_SPLINE:
            self.model = self.cubic_spline_natural_1d
        elif type == ModelType1D.LINEAR_EXTRAP:
            # TODO: replace with linear_curve_fit, a geometrical curve fitting rather than 
            # average-velocity-based method
            self.model = self.linear_avg_velocity_1d
            # self.model = self.linear_curve_fit
        elif type == ModelType1D.LINEAR_FIT:
            self.model = self.linear_curve_fit
        else:
            raise Exception("Unsupported model type")
        self.ready = False
        self.curve_points = None
        self.curve_times = None

    def rememebr(self, pose: list):
        self.buffer.storeData(pose)
        self.ready = self.buffer.isEstBufReady
    
    def predict(self, t):
        self.curve_times, points = self.buffer.get()
        T = type(t)
        if T == float or T == np.float32 or T == np.float64 or T == np.float128:
            return self.model(self.curve_times, points, t=t)
        elif T == list or T == np.ndarray:
            return self.model(self.curve_times, points, new_times=t)
        else:
            raise Exception("Unsupported value type for t parsed to function 'predict'")
    
    def linear_curve_fit(self, ts, _pos_buff, new_times=None, t=None):
        ## normalize all time wrt zero for curve fitting
        t0 = ts[0]
        ts_ = np.array([t-t0 for t in ts])
        if t is None:
            t_ = None
        else:
            t_ = t - t0 
        if new_times is None:
            new_times_ = None
        else:
            new_times_ = np.array([t-t0 for t in new_times])

        fitted_path = optimal_fit(ts_, _pos_buff)
        self.curve_points = fitted_path.reshape(fitted_path.shape[0], 1)
        #print("inputs: ------------ ")
        #print(ts_)
        #print(_pos_buff)
        #print(new_times_)
        return self.linear(ts_, fitted_path, new_times_, t_)
    
    def linear(self, ts, _pos_buff, new_times=None, t=None):#,_interp_factor=1):
        ## calculate velocities for each consecutive pair of buffered target positions (used to 
        ## estimate a current velocity for the target)
        v = (_pos_buff[-1] - _pos_buff[-2]) / (ts[-1] - ts[-2])
        if not new_times is None:
            outs = []
            #print("outputs: -------------- ")
            for n in new_times:
                dt = n - ts[-1]
                #print(dt)
                outs.append(_pos_buff[-1] + v*dt)
                # outs.append(_pos_buff[-1] + _interp_factor*v*dt)
            #print(v)
            #print(outs)
            return outs  
        elif not t is None:
            dt = t - ts[-1]
            out = _pos_buff[-1] + v*dt
            # out = _pos_buff[-1] + _interp_factor*v*dt
            #print("outputs: -------------- ")
            #print(v)
            #print(dt)
            #print(out)
            return out
        else:
            raise Exception("One of 'new_times' or 't' must be set")

    def linear_avg_velocity_1d(self, ts, _pos_buff, new_times=None, t=None):#,_interp_factor=1):
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
        # provide a virtual previous point on the current line for visualizations
        virtual_prev = _pos_buff[-1] - v*2*dt
        if not new_times is None:
            outs = []
            for n in new_times:
                dt = n - ts[-1]
                outs.append(_pos_buff[-1] + v*dt)
                # outs.append(_pos_buff[-1] + _interp_factor*v*dt)
            self.curve_points = np.array([[virtual_prev], [outs[-1]]])
            return outs 
        elif not t is None:
            dt = t - ts[-1]
            out = _pos_buff[-1] + v*dt
            # out = _pos_buff[-1] + _interp_factor*v*dt
            self.curve_points = np.array([[virtual_prev], [out]])
            return out
        else:
            raise Exception("One of 'new_times' or 't' must be set")
    
    def cubic_spline_natural_1d(self, ts, vals, new_times=None, t=None):
        return self.output_model(sp.interpolate.CubicSpline(ts, vals, bc_type='natural'), new_times, t)
    
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



