import collections

def polygon_to_point(poly: list):
    l = len(poly)
    x_sum = 0
    y_sum = 0
    for pt in poly:
        x_sum += pt[0]
        y_sum += pt[1]
    return [x_sum/l, y_sum/l]
        

class AreaPath:
    def __init__(self, buf_len):
        self.polygon_buffer = collections.deque(maxlen = buf_len)
        