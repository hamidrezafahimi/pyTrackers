import numpy as np
import pandas as pd
import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.append(pth)
from path_tracker.handy_libs import plot_now, get_file_list, get_module_function_names
from path_tracker.src import eval_model
import sys
from pathlib import Path
import src.models.functions as lm

# For all the functions in the following module:
functions = get_module_function_names(pth + "/path_tracker/src/models/functions.py")
# For all the pose data in the following directory:
files = get_file_list("/home/hamid/w/DATA/plot", ".txt")
# This program calculates the mean error for predicting n steps forward and compares 
# them in a plot

if __name__ == '__main__':
    datum = []
    for f in files:
        datum.append(np.array(pd.read_csv(f, header=None)))
    data_for_final_plot = []
    for func in functions:
        all_mean_errors = None
        for k, data in enumerate(datum):
            mean_errors = eval_model(data, getattr(lm, func))
            if all_mean_errors is None:
                all_mean_errors = np.array(mean_errors)
            else:
                all_mean_errors = np.vstack((all_mean_errors, mean_errors))
            print("Evaluating data {:s} with model {:s} done".format(Path(files[k]).stem, func))
        mean_errors_for_model = [np.mean(all_mean_errors[:,k]) for k in\
                                 range(all_mean_errors.shape[1])]
        data_for_final_plot.append([range(len(mean_errors_for_model)), 
                                    mean_errors_for_model, func])
    plot_now(plot=data_for_final_plot, xl="Number of steps forward to be predicted",
             yl="prediction error (meter)")
