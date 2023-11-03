import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()) + "/../..")
from handy_libs import calc_eucl_dist

def eval_model_1d(data, function):
    l = len(data)
    buf_len = 40
    pred_len = 50
    errors = None
    for k in range((buf_len-1), l-pred_len):
        ts = data[k-(buf_len-1):k+1,0]
        vals = data[k-(buf_len-1):k+1,1]
        new_times = data[k:k+(pred_len-1),0]
        pred_true_vals = data[k:k+(pred_len-1),1]
        try:
            new_vals = function(ts, vals, new_times=new_times)
        except:
            continue
        error = abs(new_vals - pred_true_vals)
        if errors is None:
            errors = np.array(error)
        else:
            errors = np.vstack((errors, error))
        # plotNow(plot=[[ts,vals],
        #               [data[:,0], data[:,1]],
        #               [new_times,pred_true_vals],
        #               [new_times,new_vals]], 
        #         pause=0.5)
        # print(error)
        # plotNow(plot=[[new_times, error]], pause=0.5)
    return [np.mean(errors[:,k]) for k in range(errors.shape[1])]

def eval_model(data, function):
    l = len(data)
    buf_len = 40
    pred_len = 50
    errors = None
    for k in range((buf_len-1), l-pred_len):
        times = data[k-(buf_len-1):k+1,0]
        points = data[k-(buf_len-1):k+1,1:3]
        new_times = data[k:k+(pred_len-1),0]
        pred_true_points = data[k:k+(pred_len-1),1:3]
        try:
            pred_points = function(times, points, new_times=new_times)
        except ValueError as e:
            if str(e) == "`x` must be strictly increasing sequence.":
                continue
            else:
                raise Exception(ValueError)
        error = calc_eucl_dist(pred_points, pred_true_points)
        if errors is None:
            errors = np.array(error)
        else:
            errors = np.vstack((errors, error))
    return [np.mean(errors[:,k]) for k in range(errors.shape[1])]

