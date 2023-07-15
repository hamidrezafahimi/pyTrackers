# ------ front ------

path_to_data1 = "/media/hamid/Data/BACKUPED/tcs-9-3/data/tello_test/2022-03-10/16-16-18/objectPoses_flawed.csv"
x_col_idx_1 = 1
t_col_idx_1 = 2
label_1 = "truth"

# path_to_data2 = "/media/hamid/Data/BACKUPED/tcs-9-3/data/tello_test/2022-03-10/16-16-18/predictions.csv"
# x_col_idx_2 = 1
# t_col_idx_2 = 0
# label_2 = "prediction"

# path_to_data1 = "/media/hamid/Data/BACKUPED/tcs-9-3/data/tello_test/2022-03-10/16-16-18/predictions.csv"
# x_col_idx_1 = 0
# t_col_idx_1 = 1
# label_1 = "prediction"

# ------ back ------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data1 = np.array(pd.read_csv(path_to_data1, header=None))
# data2 = np.array(pd.read_csv(path_to_data2, header=None))

t1 = data1[:,t_col_idx_1]
x1 = data1[:,x_col_idx_1]

# x2 = data2[:,x_col_idx_2]
# t2 = data2[:,t_col_idx_2]

plt.plot(t1, x1, label=label_1)
# plt.plot(t2, x2, label=label_2)
plt.legend()
plt.grid(True)
plt.show()
