# ------ front ------

path_to_data1 = "/media/hamid/Data/BACKUPED/tcs-9-3/data/tello_test/2022-03-10/16-16-18/objectPoses.csv"
x_col_idx_1 = 2
t_col_idx_1 = 1
label_1 = "truth"

# ------ back ------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data1 = np.array(pd.read_csv(path_to_data1, header=None))

t1 = data1[:,t_col_idx_1]
x1 = data1[:,x_col_idx_1]

plt.plot(t1, x1, label=label_1)
plt.legend()
plt.grid(True)
plt.show()
