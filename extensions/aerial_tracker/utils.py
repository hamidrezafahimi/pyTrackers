
import numpy as np

def calc_ground_range(alt, range):
    assert(alt >= 0 and range >= 0)
    ang = np.arccos(alt/range)
    return np.arctan(ang) * alt