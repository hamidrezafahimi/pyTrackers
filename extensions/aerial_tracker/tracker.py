import sys
from pathlib import Path
root_path = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(0, root_path)
from extensions.camera_kinematics import CameraKinematics
from matplotlib import pyplot as plt
from .utils import calc_ground_range
import cv2 as cv
import numpy as np

class AerialTracker(CameraKinematics):
    def __init__(self, wps, vr, map_width, map_height, cx, cy, f=None, w=None, h=None, hfov=None):
        self._visualRange = vr
        self._wps = wps
        self.__mapWidth = map_width
        self.__mapHeight = map_height
        super().__init__(self._wps[0], cx, cy, f, w, h, hfov)
        # self.ax = plt.figure().add_subplot(111, projection='3d')
        # self.ax.view_init(elev=-140, azim=-60)
        self.__initNormalization()
        self.map = np.zeros((self.__mapWidth, self.__mapHeight))
        # self.initGlobalMap()
    
    def predict(self, imu_meas, cam_ps):
        # self.ax.cla()
        pos1, pos2, pos3, pos4 = self.get_camera_fov_area(imu_meas, cam_ps)
        cv.circle(self.map, self.__NED2IMG(pos1), 2, (255, 255, 255), 2)
        cv.circle(self.map, self.__NED2IMG(pos2), 2, (255, 255, 255), 2)
        cv.circle(self.map, self.__NED2IMG(pos3), 2, (255, 255, 255), 2)
        cv.circle(self.map, self.__NED2IMG(pos4), 2, (255, 255, 255), 2)
        cv.imshow("map", self.map)
        # self.ax.scatter(pos1[0], pos1[1])
        # self.ax.scatter(pos2[0], pos2[1])
        # self.ax.scatter(pos3[0], pos3[1])
        # self.ax.scatter(pos4[0], pos4[1])
        # plt.pause(0.02)

    def __initNormalization(self):
        minX = 1e6
        minY = 1e6
        maxX = -1e6
        maxY = -1e6
        for wp in self._wps:
            r = calc_ground_range(wp[2], self._visualRange)
            dist_x_high = wp[0] + r
            dist_x_low = wp[0] - r
            dist_y_high = wp[1] + r
            dist_y_low = wp[1] - r
            if minX > dist_x_low:
                minX = dist_x_low
            if maxX < dist_x_high:
                maxX = dist_x_high
            if minY > dist_y_low:
                minY = dist_y_low
            if maxY < dist_y_high:
                maxY = dist_y_high
        assert(maxX > minX and maxY > minY)
        self.__mppr_x = (maxX - minX) / self.__mapWidth
        self.__mppr_y = (maxY - minY) / self.__mapHeight
        self.__minX = minX
        self.__minY = minY
    
    def __NED2IMG(self, pnt):
        inertia_x, inertia_y, _ = pnt
        return (int((inertia_x - self.__minX) / self.__mppr_x), 
                int((inertia_y- self.__minY) / self.__mppr_y))