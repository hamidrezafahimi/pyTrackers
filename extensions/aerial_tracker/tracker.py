import sys
from pathlib import Path
root_path = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(0, root_path)
from extensions.camera_kinematics import CameraKinematics
from lib.utils.geom import gps_to_ned
from matplotlib import pyplot as plt
from .utils import calc_ground_range, make_panorama_scan, predict_linear_probs, NED2IMG_array, NED2IMG_single
import cv2 as cv
import numpy as np
import time
import copy

## NOTE:
# The geographic directions in the map image:
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
        self.pan_width = 1080
        self.show_demo = True
        # self.initGlobalMap()
        self.v_std_dev = 0.5
        self.beta_std_dev = 10 * np.pi / 180
        self.samples_num = 100
        self.object_pose_buffer = np.array([])
        self.object_pose_buffer_len = 30
        self.const_dt = 0.2
        self.object_points = []

    def scan(self, imu_meas, cam_ps):
        pos1, pos2, pos3, pos4 = self.get_camera_fov_area(imu_meas, cam_ps)
        pnt1 = NED2IMG_single(pos1[0], pos1[1], self.__minX, self.__minY, self.__mppr)
        pnt2 = NED2IMG_single(pos2[0], pos2[1], self.__minX, self.__minY, self.__mppr)
        pnt3 = NED2IMG_single(pos3[0], pos3[1], self.__minX, self.__minY, self.__mppr)
        pnt4 = NED2IMG_single(pos4[0], pos4[1], self.__minX, self.__minY, self.__mppr)
        print("pos before: ", pos1)
        print("pos after: ", pnt1)
        cv.circle(self.map, pnt1, 2, (255, 255, 255), 2)
        cv.circle(self.map, pnt2, 2, (255, 255, 255), 2)
        cv.circle(self.map, pnt3, 2, (255, 255, 255), 2)
        cv.circle(self.map, pnt4, 2, (255, 255, 255), 2)


        cam_ned_loc = self.get_cam_pos_ned(cam_ps)
        print("cam_ned_loc: ", cam_ned_loc)
        cam_pix_loc = NED2IMG_single(cam_ned_loc[0], cam_ned_loc[1], self.__minX, self.__minY, self.__mppr)
        cv.circle(self.map, cam_pix_loc, 2, (255, 255, 255), 2)
        cv.circle(self.map, cam_pix_loc, self.min_r_pix, (255, 255, 255), 1)
        cv.circle(self.map, cam_pix_loc, self.max_r_pix, (255, 255, 255), 1)
        
        map_shape = self.map.shape
        min_local_y_pix = max(cam_pix_loc[1] - self.max_r_pix, 0)
        min_local_x_pix = max(cam_pix_loc[0] - self.max_r_pix, 0)
        max_local_y_pix = min(cam_pix_loc[1] + self.max_r_pix, map_shape[0])
        max_local_x_pix = min(cam_pix_loc[0] + self.max_r_pix, map_shape[1])
        local_map = self.map[min_local_y_pix:max_local_y_pix, min_local_x_pix:max_local_x_pix]
        pan_scan = make_panorama_scan(local_map, self.pan_width, self.min_r_pix/self.max_r_pix, (600, 200))
        cv.rectangle(self.map, (min_local_x_pix, min_local_y_pix), (max_local_x_pix, max_local_y_pix), (255, 0, 0), 1)
        if self.show_demo:
            cv.imshow("global map", self.map)
            temp = cv.resize(local_map, (900, 900), interpolation=cv.INTER_LINEAR)
            #cv.imshow("local map", temp)
            #cv.imshow("local map", local_map)
            #cv.imshow("pan scan", pan_scan)
        return pan_scan

    def updateMap(self):
        # TODO: Only render the area which is to be cropped
        newMap = np.zeros((self.__mapHeight_pix, self.__mapWidth_pix, 1), np.uint8)
        for pt in self.object_points:
            cv.circle(newMap, (pt[0], pt[1]), 2, 127, 2)
        prob_points_ned = predict_linear_probs(self.object_pose_buffer[:,0], self.object_pose_buffer[:,1], self.object_pose_buffer[:,2], self.const_dt, self.v_std_dev, self.beta_std_dev, self.samples_num)
        prob_points_pix = NED2IMG_array(prob_points_ned, self.__minX, self.__minY, self.__mppr)
        l = prob_points_ned.shape[0]
        for k in range(l):
            j, i = prob_points_pix[k, 0], prob_points_pix[k, 1] 
            if newMap[i,j] < 255:
                newMap[i, j] += 1
        newMap = cv.equalizeHist(newMap)
        print(self.object_points)
        pntx = NED2IMG_single(-15, 6, self.__minX, self.__minY, self.__mppr)
        print("pntx: ", pntx)
        # cv.circle(newMap, pntx, 2, (255, 255, 255), 10)
        self.map = newMap.copy()

    def predict(self, imu_meas, cam_ps, object_pose, score_map=None):
        # buffer last object poses
        if not object_pose is None:
            print("obj before: ", object_pose)
            pt = NED2IMG_single(object_pose[1], object_pose[2], self.__minX, self.__minY, self.__mppr)
            print("obj after: ", pt)
            self.object_points.append(pt)

        if object_pose is None:
            pass
        elif self.object_pose_buffer.shape[0] == 0:
            self.object_pose_buffer = np.array([[object_pose[0], object_pose[1], object_pose[2]]])
        elif self.object_pose_buffer.shape[0] <= self.object_pose_buffer_len:
            self.object_pose_buffer = np.vstack([self.object_pose_buffer, np.array([object_pose[0], object_pose[1], object_pose[2]])])
        else:
            self.object_pose_buffer = np.delete(self.object_pose_buffer, (0), axis=0)
            self.object_pose_buffer = np.vstack([self.object_pose_buffer, np.array([object_pose[0], object_pose[1], object_pose[2]])])
        
        if self.object_pose_buffer.shape[0] <= 1:
            return None
        
        # to avoid accumulation of drawings:
        #self.map = np.zeros((self.__mapWidth_pix, self.__mapHeight_pix), dtype=np.uint8)
        # TODO: The followings must be replaced with a function named 'updateMap'
        #self.map = cv.imread("/home/hamid/w/REPOS/pyTrackers/launch/track/tes.png", cv.IMREAD_GRAYSCALE)
        #self.map = cv.resize(self.map, (self.__mapHeight_pix, self.__mapWidth_pix), interpolation=cv.INTER_LINEAR)
        self.updateMap()
        pan_scan = self.scan(imu_meas, cam_ps)
        # self.getOptimalROI(score_map, pan_scan)

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
        print("map dims: ", self.__mapWidth_pix, self.__mapHeight_pix) 
        # assert (self.__mppr_x == self.__mppr_y), \
                #"the map must be define as a square, not a rectangle (width: {}, height:{})".format(maxY - minY, maxX - minX)
    