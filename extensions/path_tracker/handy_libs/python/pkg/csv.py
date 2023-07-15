import pandas as pd
import numpy as np

def write2csv(inp_dict, directory):
    df = pd.DataFrame(inp_dict)
    df.to_csv(directory, mode='w', index=False, header=False)

def read_csv_array(path_to_data):
    return np.array(pd.read_csv(path_to_data, header=None))