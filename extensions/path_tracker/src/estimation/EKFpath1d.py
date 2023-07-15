import numpy as np
from path_tracker.lib.models.modelling import ModelType1D, Modeler1D

model = Modeler1D(type = ModelType1D.CUBIC_SPLINE)

def ekf_estimator_1d(x_pred, t, P_pred, dt, measurement=None):
    x_pred = np.array([x_pred])
    # Initialize state vector and covariance matrix
    if not measurement is None:
        measurement = np.array([measurement])
    #     # x_pred = np.array([x])
    #     # P_pred = P
    # else:
        # Measurement update
        H = np.array([[1]])
        R = np.array([[0.01]])
        y = measurement - x_pred
        # P_pred : P_{k-1}^-
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        # K : K_k
        # x_pred : x_{k-1}^-
        x_pred = x_pred + K @ y
        # x_pred : x_k^+
        P_pred = (np.eye(1) - K @ H) @ P_pred
        # P_pred : P_k^+
        current_pose = np.array([x_pred[0], t])
        model.storeData(current_pose)    

    if model.ready:
        # Time update
        F = np.eye(1)
        Q = np.array([[1]])
        x_pred = np.array([model.predict(t+dt)]) 
        # x_pred : x_k^-
        F[0,0] = (model.predict(t+dt) - model.predict(t)) / dt
        P_pred = F @ P_pred @ F.T + Q
        # P_pred : P_k^-
        return x_pred[0], P_pred
    else:
        return x_pred[0], P_pred
