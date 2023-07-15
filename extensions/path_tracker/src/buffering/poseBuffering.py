import collections
import numpy as np

class PoseBuffer:
    def __init__(self, buf_len):
        self.bufLen = buf_len
        self.x = collections.deque(maxlen = buf_len)
        self.y = collections.deque(maxlen = buf_len)
        self.z = collections.deque(maxlen = buf_len)
        self.t = collections.deque(maxlen = buf_len)
        self.isFull = False
    
    def updateIsFull(self):
        if not self.isFull:
            self.isFull = (len(self.t) == self.bufLen)

    def getX(self):
        return np.array(self.x)
    
    def getY(self):
        return np.array(self.y)
    
    def getZ(self):
        return np.array(self.z)
    
    def getT(self):
        return np.array(self.t)

    def getNearestX(self, _t,  not_this_t=-np.Inf):
        return self.getNearest(self.x, self.t, _t)
    
    def getNearest(self, vals, times, _t):
        minDist = 1e6
        idx = None
        for k, num in enumerate(vals):
            dist = abs(times[k] - _t)
            if minDist > dist:
                minDist = dist
                idx = k
        return vals[idx], times[idx]


class Pose1DBuffer(PoseBuffer):
    def __init__(self, buf_len):
        super().__init__(buf_len)

    def buffer(self, pose):
        self.x.append(pose[0])
        self.t.append(pose[1])
        super().updateIsFull()


class Pose2DBuffer(PoseBuffer):
    def __init__(self, buf_len):
        super().__init__(buf_len)

    def buffer(self, pose):
        self.x.append(pose[0])
        self.y.append(pose[1])
        self.t.append(pose[2])
        super().updateIsFull()


class Pose3DBuffer(PoseBuffer):
    def __init__(self, buf_len):
        super().__init__(buf_len)

    def buffer(self, pose):
        self.x.append(pose[0])
        self.y.append(pose[1])
        self.z.append(pose[2])
        self.t.append(pose[3])
        super().updateIsFull()
