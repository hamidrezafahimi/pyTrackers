from enum import Enum
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()) + "/..")
from buffering.pathBuffering import PathBufferer1D
from .functions import linear_delayed, linear
from .utils import cubic_spline_natural_1d, linear_avg_velocity_1d
import numpy as np

class ModelType1D(Enum):
    LINEAR = 1
    CUBIC_SPLINE = 2

class Modeler1D:
    def __init__(self, type: ModelType1D):
        self.buffer = PathBufferer1D()
        if type == ModelType1D.CUBIC_SPLINE:
            self.model = cubic_spline_natural_1d
        elif type == ModelType1D.LINEAR:
            # self.model = linear_delayed
            # self.model = linear
            self.model = linear_avg_velocity_1d
        else:
            raise Exception("Unsupported model type")
        self.ready = False

    def rememebr(self, pose: list):
        self.buffer.storeData(pose)
        self.ready = self.buffer.isEstBufReady
    
    def predict(self, t):
        ts, points = self.buffer.get()
        T = type(t)
        if T == float or T == np.float32 or T == np.float64 or T == np.float128 or\
           T == np.float256:
            return self.model(ts, points, t=t)
        elif T == list or T == np.ndarray:
            return self.model(ts, points, new_times=t)
        else:
            raise Exception("Unsupported value type for t parsed to function 'predict'")
