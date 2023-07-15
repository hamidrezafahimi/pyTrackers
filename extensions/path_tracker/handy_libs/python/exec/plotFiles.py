import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/.."
sys.path.insert(1, pth)

import pkg.argparse as al
import pkg.csv as cl

# import pandas as pd
import matplotlib.pyplot as plt

files, names = al.get_parsed_files()

x_col_idx = 2
t_col_idx = 1

for k, file in enumerate(files):
    data = cl.read_csv_array(file)
    t = data[:, t_col_idx]
    x = data[:, x_col_idx]
    plt.plot(t, x, label=names[k])
    plt.scatter(t, x)
    plt.scatter(t[0], x[0], color='r')
plt.grid(True)
plt.legend()
plt.show()
    