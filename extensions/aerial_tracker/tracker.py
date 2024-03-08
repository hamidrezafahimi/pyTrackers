import sys
from pathlib import Path
root_path = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(0, root_path)
from extensions.camera_kinematics import CameraKinematics
from lib.utils.geom import gps_to_ned
from matplotlib import pyplot as plt
from .utils import calc_ground_range
import cv2 as cv
import numpy as np
import time
import copy

## NOTE:
# Here are the geographic directions in the map image:
# --> N
# |
# v
# E


class AerialTracker(CameraKinematics):
    def __init__(self, wps, mppr, cx, cy, vr=100, f=None, w=None, h=None, hfov=None):
        super().__init__(copy.deepcopy(wps[0]), cx, cy, f, w, h, hfov)
        self._visualRange = vr
        self._wps = []
        for k in range(wps.shape[0]):
            self._wps.append([*gps_to_ned(self.ref_loc, wps[k])])
        #self.__mapWidth = map_width
        #self.__mapHeight = map_height
        self.__mppr = mppr
        # self.ax = plt.figure().add_subplot(111, projection='3d')
        # self.ax.view_init(elev=-140, azim=-60)
        self.min_r_meter = 3
        self.max_r_meter = 35
        self.__initNormalization()
        # self.initGlobalMap()
    
    def predict(self, imu_meas, cam_ps):
        # to avoid accumulation of drawings:
        self.map = np.zeros((self.__mapWidth_pix, self.__mapHeight_pix))
        
        pos1, pos2, pos3, pos4 = self.get_camera_fov_area(imu_meas, cam_ps)
        cv.circle(self.map, self.__NED2IMG(pos1), 2, (255, 255, 255), 2)
        cv.circle(self.map, self.__NED2IMG(pos2), 2, (255, 255, 255), 2)
        cv.circle(self.map, self.__NED2IMG(pos3), 2, (255, 255, 255), 2)
        cv.circle(self.map, self.__NED2IMG(pos4), 2, (255, 255, 255), 2)

        cam_ned_loc = self.get_cam_pos_ned(cam_ps)
        cam_pix_loc = self.__NED2IMG(cam_ned_loc)
        cv.circle(self.map, cam_pix_loc, 2, (255, 255, 255), 2)
        cv.circle(self.map, cam_pix_loc, self.min_r_pix, (255, 255, 255), 1)
        cv.circle(self.map, cam_pix_loc, self.max_r_pix, (255, 255, 255), 1)
        cv.imshow("map", self.map)

    def __initNormalization(self):
        minX = 1e6
        minY = 1e6
        maxX = -1e6
        maxY = -1e6
        for wp in self._wps:
            r = calc_ground_range(abs(wp[2]), self._visualRange)
            dist_x_high = wp[0] + r
            dist_x_low = wp[0] - r
            dist_y_high = wp[1] + r
            dist_y_low = wp[1] - r
            # print(wp)
            # print(self._visualRange)
            # print(r)
            # print(dist_x_high, dist_x_low, dist_y_high, dist_y_low)
            # time.sleep(2)
            # print(".......")
            if minX > dist_x_low:
                minX = dist_x_low
            if maxX < dist_x_high:
                maxX = dist_x_high
            if minY > dist_y_low:
                minY = dist_y_low
            if maxY < dist_y_high:
                maxY = dist_y_high
        assert (maxX > minX and maxY > minY), "min may not be more than max"
        #self.__mppr_x = (maxX - minX) / self.__mapWidth
        #self.__mppr_y = (maxY - minY) / self.__mapHeight
        self.__mapWidth_pix = int((maxX - minX) / self.__mppr)
        self.__mapHeight_pix = int((maxY - minY) / self.__mppr)
        self.__minX = minX
        self.__minY = minY
        self.map = np.zeros((self.__mapWidth_pix, self.__mapHeight_pix))
        self.min_r_pix = int(self.min_r_meter / self.__mppr)
        self.max_r_pix = int(self.max_r_meter / self.__mppr)
        # assert (self.__mppr_x == self.__mppr_y), \
                #"the map must be define as a square, not a rectangle (width: {}, height:{})".format(maxY - minY, maxX - minX)
    
    def __NED2IMG(self, pnt):
        inertia_x, inertia_y, _ = pnt
        pixel_x = int((inertia_x - self.__minX) / self.__mppr)
        pixel_y = int((inertia_y - self.__minY) / self.__mppr)
        # print(inertia_x, self.__minX, pixel_x)
        # print(inertia_y, self.__minY, pixel_y)
        # print("--------")
        # time.sleep(1)
        return (pixel_x, pixel_y)
    
    def __meter2pix_dist(self, meter_dist):
        pass
