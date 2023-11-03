import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()) + "/..")
from src import EKFEstimator

class AerialObserver:
    def __init__(self):
        self.estimator = EKFEstimator()
        self.tracking_state = []
    
    def doPrediction(self, y, t, t_next):
        self.tracking_state.append(not y is None)
        if sum(self.tracking_state[-11:]) == 11:
            print('good')
            return self.estimator.update(y, t, t_next)
        else:
            print('bad')
            return self.estimator.update(None, t, t_next)
        
        