import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()) + "/..")
from .modelling import ModelType1D, Modeller1D
import numpy as np
import time
from .stochastic import predict_pose_probs

class EKFEstimator:
    def __init__(self):
        # self.model_x = Modeler1D(type = ModelType1D.CUBIC_SPLINE)
        # self.model_y = Modeler1D(type = ModelType1D.CUBIC_SPLINE)
        #self.model_x = Modeller1D(type = ModelType1D.LINEAR_EXTRAP)
        #self.model_y = Modeller1D(type = ModelType1D.LINEAR_EXTRAP)
        self.model_x = Modeller1D(type = ModelType1D.LINEAR_OPTIMAL_FIT)
        self.model_y = Modeller1D(type = ModelType1D.LINEAR_OPTIMAL_FIT)
        self.P = np.eye(2)
        self.x = np.array([[0],[0]])	
        self.H = np.eye(2)
        self.R = np.eye(2)
        self.Q = np.array([[1], [1]])
        self.F = np.eye(2)
        v_std_dev = 1  # Velocity standard deviation
        beta_std_dev = 30 * np.pi / 180  # Beta (angle) standard deviation
        K = 200
        self.gen = predict_pose_probs(None, v_std_dev, beta_std_dev, K)
        next(self.gen)

    def update(self, y, t, dt):
        #print("a call to update =============== ")
        #time.sleep(3)
        if not y is None:
            self.model_x.rememebr([y[0], t])
            self.model_y.rememebr([y[1], t])
            y = [y[0], y[1]] #TODO: Remove this after adding 3d pose estimation feature

        self.estimate(t, dt, y)

        if y is None:
            nxpt = [t+dt, self.x[0][0], self.x[1][0]]
            data = (None, nxpt)
        elif not self.model_x.curve_times is None and self.model_x.curve_times.shape[0] >= 2:
            ## Indexing such that 'old_point' and 'new_point' are lists of scalars rather than nested np arrays
            old_point = [self.model_x.curve_times[-2], self.model_x.curve_points[-2][0], self.model_y.curve_points[-2][0]]
            new_point = [self.model_x.curve_times[-1], self.model_x.curve_points[-1][0], self.model_y.curve_points[-1][0]]
            data = ((old_point, new_point), None)
        else:
            return None, None, None

        # if self.model_x.ready:
        #     return [self.x[0][0], self.x[1][0]], \
        #         np.hstack([self.model_x.curve_points, self.model_y.curve_points]), \
        #         self.model_x.curve_times
        # else:
        #     return None, None, None
        prob_points = self.gen.send(data)
        return prob_points, \
            np.hstack([self.model_x.curve_points, self.model_y.curve_points]), \
            self.model_x.curve_times


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
            x_preds, switched_x = self.model_x.predict([t, t+dt])
            y_preds, switched_y = self.model_y.predict([t, t+dt])
            if (switched_y or switched_x):
                print("switched model ----- resetting covariance matrix")
                self.P = np.eye(2)
            # Time update
            self.x = np.array([[x_preds[1]], [y_preds[1]]]) 
            # self.x : x_k^-
            #self.F[0,0] = (self.model_x.predict(t+dt) - self.model_x.predict(t)) / dt
            #self.F[1,1] = (self.model_y.predict(t+dt) - self.model_y.predict(t)) / dt
            self.F[0,0] = (x_preds[1] - x_preds[0]) / dt
            self.F[1,1] = (y_preds[1] - y_preds[0]) / dt
            self.P = (self.F @ self.P @ self.F.T + self.Q)
            # print(self.P)
            # self.P : P_k^-

