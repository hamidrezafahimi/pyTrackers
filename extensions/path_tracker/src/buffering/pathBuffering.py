from .poseBuffering import Pose1DBuffer
import numpy as np

class PathBufferer1D:
    def __init__(self, record_buf_len=200, estimate_buf_len=20):
        self.recordBuffer = Pose1DBuffer(record_buf_len)
        self.lastEstBufTime = None
        self.recBufLen = record_buf_len
        self.estBufLen = estimate_buf_len
        self.estimateBuffer = np.zeros((self.estBufLen, 2))
        self.maxTimeStep = 0.5
        self.minTimeStep = 0.01
        self.isEstBufReady = False
        self.lastEstBufUpdate = None

    def storeData(self, pose):
        self.recordBuffer.buffer(pose)
        self.updateEstimateBuffer(pose[1])

    def updateEstimateBuffer(self, current_t):
        minT = min(self.recordBuffer.t)
        maxT = max(self.recordBuffer.t)
        timeStep = min((maxT - minT) / self.estBufLen, self.maxTimeStep)
        if timeStep < self.minTimeStep:
            return
        maxTIdx = np.argmax(self.recordBuffer.t)
        maxTVal = self.recordBuffer.x[maxTIdx]
        self.estimateBuffer[self.estBufLen-1,:] = [maxTVal, maxT]
        t = maxT - timeStep
        for k in range(self.estBufLen-1):
            val = np.interp(t, self.recordBuffer.t, self.recordBuffer.x)
            self.estimateBuffer[self.estBufLen-(k+2),:] = [val, t]
            t -= timeStep
        self.isEstBufReady = True

    def get(self):
        if not self.isEstBufReady:
            return None, None
        return self.estimateBuffer[:,1], self.estimateBuffer[:,0]

