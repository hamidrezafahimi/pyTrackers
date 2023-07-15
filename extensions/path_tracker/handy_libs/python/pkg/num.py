import numpy as np

def calc_eucl_dist(a, b):
    if isinstance(a[0], list) and isinstance(b[0], list):
        if len(a) != len(b):
            raise Exception("Length of two lists must be the same")
        return calc_array_eucl_dist(a, b)
    
    elif isinstance(a[0], (np.ndarray, np.generic)) or isinstance(b[0], (np.ndarray, np.generic)):
        # if a.shape[1] != b.shape[1]:
        #     raise Exception("Length of two np arrays must be the same")
        return calc_array_eucl_dist(a, b)
        
    else:
        return calc_point_eucl_dist(a, b)
        
def calc_array_eucl_dist(a, b):
    out = []
    for k, pt in enumerate(a):
        out.append(calc_point_eucl_dist(pt, b[k]))
    return np.array(out)

def calc_point_eucl_dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)