import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()) + "/..")
from enum import Enum
from src import EKFEstimator

class VOTPathObserver:
    def __init__(self):
        self.estimator = EKFEstimator()
    
    def doPrediction(self, y, t, t_next):
        return self.estimator.update(y, t, t_next)
        
        