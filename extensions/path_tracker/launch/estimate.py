import numpy as np
import pandas as pd
import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(1, pth)
from path_tracker.lib import EKFEstimator

estimator = EKFEstimator()

p_pred = np.eye(1)

data = pd.read_csv("/home/hamid/d/BACKUPED/tcs-9-3/data/VIOT/park_mavic_1/new/park_mavic_1_target_poses.txt", header=None)
tags = pd.read_csv("/home/hamid/d/BACKUPED/tcs-9-3/data/VIOT/park_mavic_1/new/MIXFORMER_park_mavic_1_viot_params.txt", header=None)

data = np.array(data)
tags = np.array(tags)

t_predicted = []
x_predicted = []
y_predicted = []
idx_prediction = []

l = len(data)
x_ = 1
gt_x = []
gt_y = []
gt_t = []
for k in range(1, l-1):
# for k in range(1, l-50):
# for k in range(1, l-60):
    t = data[k][0]
    next_t = data[k+1][0]
    idx = data[k][4]
    if (tags[k-1][3] == 0):
        y = None
    else:
        y = [data[k][1], data[k][2]]
        gt_x.append(y[0])
        gt_y.append(y[1])
        gt_t.append(t)
    x_ = estimator.update(y, t, next_t)
    if not x_[0] is None:
        t_predicted.append(next_t)
        x_predicted.append(x_[0])
        y_predicted.append(x_[1])
        idx_prediction.append(idx+1)

df = pd.DataFrame({'Time':t_predicted, 'Xs':x_predicted, 'Ys':y_predicted, 'related idx':idx_prediction})
gt = pd.DataFrame({'Time':gt_t, 'Xs':gt_x, 'Ys':gt_y})
df.to_csv("/home/hamid/d/BACKUPED/tcs-9-3/data/VIOT/park_mavic_1/new/predictions.csv", mode='w', index=False, header=False)
gt.to_csv("/home/hamid/d/BACKUPED/tcs-9-3/data/VIOT/park_mavic_1/new/gt.csv", mode='w', index=False, header=False)

