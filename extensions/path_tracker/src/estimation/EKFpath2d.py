import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()) + "/..")
from models.modelling import ModelType1D, Modeler1D
import numpy as np

class EKFEstimator:
    def __init__(self):
        # self.model_x = Modeler1D(type = ModelType1D.CUBIC_SPLINE)
        # self.model_y = Modeler1D(type = ModelType1D.CUBIC_SPLINE)
        self.model_x = Modeler1D(type = ModelType1D.LINEAR)
        self.model_y = Modeler1D(type = ModelType1D.LINEAR)
        # self.model_x = Modeler1D(type = ModelType1D.QUADRATIC_SPLINE)
        # self.model_y = Modeler1D(type = ModelType1D.QUADRATIC_SPLINE)
        self.P = 100 * np.eye(2)
        self.x = np.array([[0],[0]])	
        self.H = np.eye(2)
        self.R = np.eye(2)
        self.Q = np.array([[1], [1]])
        self.F = np.eye(2)

    def update(self, y, t, t_next):
        if not y is None:
            self.model_x.rememebr([y[0], t])
            self.model_y.rememebr([y[1], t])
            y = [y[0], y[1]] #TODO: Remove this after adding 3d pose estimation feature

        dt = t_next - t
        self.estimate(t, dt, y)
        if self.model_x.ready:
            return [self.x[0][0], self.x[1][0]]
        else:
            return [None, None]

    def estimate(self, t, dt, measurement=None):
        if not measurement is None:
            measurement = np.array([measurement]).T
            # Measurement update
            y = measurement - self.x
            # self.P : P_{k-1}^-
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            # K : K_k
            # self.x : x_{k-1}^-
            self.x = self.x + K @ y
            # self.x : x_k^+
            self.P = (np.eye(1) - K @ self.H) @ self.P
            # self.P : P_k^+
            # TODO: Can't activate the following due to divergence of P.
            # self.model_x.storeData([self.x[0][0], t])
            # self.model_y.storeData([self.x[1][0], t])

        if self.model_x.ready:
            # Time update
            self.x = np.array([[self.model_x.predict(t+dt)], [self.model_y.predict(t+dt)]]) 
            # self.x : x_k^-
            self.F[0,0] = (self.model_x.predict(t+dt) - self.model_x.predict(t)) / dt
            self.F[1,1] = (self.model_y.predict(t+dt) - self.model_y.predict(t)) / dt
            self.P = self.F @ self.P @ self.F.T + self.Q
            # self.P : P_k^-

