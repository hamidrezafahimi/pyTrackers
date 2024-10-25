import sys
from pathlib import Path
root_path = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(0, root_path)
from extensions.camera_kinematics import CameraKinematics
from lib.utils.geom import get_ned_wrt_ref
from matplotlib import pyplot as plt
from .utils import calc_ground_range, make_panorama_scan, predict_linear_probs, NED2IMG_array, NED2IMG_single
import cv2 as cv
import numpy as np
import time
import copy
from .mmodel import EKFEstimator
import pandas as pd

# For Midas
from extensions.midas.midas import Midas

## NOTE:
# The geographic directions in the map image:
# --> N
# |
# v
# E


class AerialTracker(CameraKinematics):
    def __init__(self, wps, mppr, cx, cy, w, h, vr=100, f=None, hfov=None):
        super().__init__(copy.deepcopy(wps[0]), cx, cy, f, w, h, hfov)
        self._visualRange = vr
        self._wps = []
        for k in range(wps.shape[0]):
            self._wps.append([*get_ned_wrt_ref(self.ref_loc, wps[k])])
        #self.__mapWidth = map_width
        #self.__mapHeight = map_height
        self.__mppr = mppr
        self.occupancy_map = np.zeros((self.__mapHeight_pix, self.__mapWidth_pix, 1), np.uint8)
        # self.ax = plt.figure().add_subplot(111, projection='3d')
        # self.ax.view_init(elev=-140, azim=-60)
        self.min_r_meter = 3
        self.max_r_meter = 35
        self.__initNormalization()
        self.pan_width = 1080
        # self.initGlobalMap()
        #self.v_std_dev = 0.5
        self.v_std_dev = 1 
        self.beta_std_dev = 20 * np.pi / 180
        #self.samples_num = 100
        self.samples_num = 300
        self.lastOptimalROI = None
        # self.object_pose_buffer = np.array([])
        # self.object_pose_buffer_len = 30
        self.const_dt = 0.2
        self.object_points_pix = []
        self.estimator = EKFEstimator()
        self.map = None
        self.extraVis = False
        self.show_demo = False
        self.doPanScan = False
        ## Homographic transform - Simulation of image capturing from probabilistic map
        self._imageCorners = np.array([[0, 0],[w-1, 0],[w-1, h-1],[0, h-1]])
        self._pixelMapSize = (w, h)
        self.iter = 0
        self.picturing_thresh = 80
        self.midas_object = Midas("midas_v21_384.pt", "midas_v21_384")

    def scan(self, imu_meas, cam_ps):
        if self.map is None:
            return None, None
        pos1, pos2, pos3, pos4 = self.get_camera_fov_area(imu_meas, cam_ps)
        pnt1 = NED2IMG_single(pos1[0], pos1[1], self.__minX, self.__minY, self.__mppr)
        pnt2 = NED2IMG_single(pos2[0], pos2[1], self.__minX, self.__minY, self.__mppr)
        pnt3 = NED2IMG_single(pos3[0], pos3[1], self.__minX, self.__minY, self.__mppr)
        pnt4 = NED2IMG_single(pos4[0], pos4[1], self.__minX, self.__minY, self.__mppr)
        in_map_corners_pix = np.array([[pnt1[0], pnt1[1]], [pnt2[0], pnt2[1]], [pnt3[0], pnt3[1]], [pnt4[0], pnt4[1]]])
        if self.extraVis:
            cv.circle(self.mapVis, pnt1, 2, (255, 255, 255), 2)
            cv.circle(self.mapVis, pnt2, 2, (255, 255, 255), 2)
            cv.circle(self.mapVis, pnt3, 2, (255, 255, 255), 2)
            cv.circle(self.mapVis, pnt4, 2, (255, 255, 255), 2)

        if self.doPanScan:
            cam_ned_loc = self.get_cam_pos_ned(cam_ps)
            cam_pix_loc = NED2IMG_single(cam_ned_loc[0], cam_ned_loc[1], self.__minX, self.__minY, self.__mppr)
            if self.extraVis:
                cv.circle(self.mapVis, cam_pix_loc, 2, (255, 255, 255), 2)
                cv.circle(self.mapVis, cam_pix_loc, self.min_r_pix, (255, 255, 255), 1)
                cv.circle(self.mapVis, cam_pix_loc, self.max_r_pix, (255, 255, 255), 1)
            
            map_shape = self.map.shape
            min_local_y_pix = max(cam_pix_loc[1] - self.max_r_pix, 0)
            min_local_x_pix = max(cam_pix_loc[0] - self.max_r_pix, 0)
            max_local_y_pix = min(cam_pix_loc[1] + self.max_r_pix, map_shape[0])
            max_local_x_pix = min(cam_pix_loc[0] + self.max_r_pix, map_shape[1])
            local_map = self.map[min_local_y_pix:max_local_y_pix, min_local_x_pix:max_local_x_pix]
            # pan_scan = make_panorama_scan(local_map, self.pan_width, self.min_r_pix/self.max_r_pix, (600, 200))
            if self.show_demo:
                if self.extraVis:
                    cv.imshow("global map", self.mapVis)
                else:
                    cv.imshow("global map", self.map)
                # temp = cv.resize(local_map, (900, 900), interpolation=cv.INTER_LINEAR)
                # cv.imshow("local map", temp)
                #cv.imshow("local map", local_map)
                # cv.imshow("pan scan", pan_scan)
            # return pan_scan, in_map_corners_pix
            return None, in_map_corners_pix
        else:
            return None, in_map_corners_pix

    
    # def updateMap(self, est_pos_ned, fitted_spline=None):
    def updateMap(self, prob_points_ned, fitted_spline=None):
        # TODO: Only render the area which is to be cropped
        self.map = self.occupancy_map.copy()
        if self.extraVis:
            self.mapVis = self.map.copy()
        
        if prob_points_ned is None:
            return

        # for k in range(prob_points_ned.shape[0]):
        #     df_marker = pd.DataFrame({'Xs':[prob_points_ned[k,1]], 'Ys':[prob_points_ned[k,2]], 'Time':[prob_points_ned[k,0]]})
        #     df_marker.to_csv("/home/hamid/sparsepoints.csv", mode='a', index=False, header=False)
            
        prob_points_pix = NED2IMG_array(prob_points_ned, self.__minX, self.__minY, self.__mppr)
        # print (prob_points_ned)
        # print (prob_points_pix)
        # print ('---------------')
        l = prob_points_ned.shape[0]
        if self.extraVis:
            for k in range(l):
                j, i = prob_points_pix[k, 0], prob_points_pix[k, 1] 
                if self.map[i,j] < 255:
                    self.map[i, j] += 1
                    self.mapVis[i, j] += 1
        else:
            for k in range(l):
                j, i = prob_points_pix[k, 0], prob_points_pix[k, 1] 
                if self.map[i,j] < 255:
                    self.map[i, j] += 1
                    
        # self.map = cv.GaussianBlur(self.map,(11,11),0)
        self.map = cv.equalizeHist(self.map)
        if self.extraVis:
            self.mapVis = cv.GaussianBlur(self.map,(11,11),0)
            self.mapVis = cv.equalizeHist(self.map)

        ## visualize the polynomial which the estimator has fitted to target's path
        if self.extraVis:
            if not fitted_spline is None:
                for k in range(1, fitted_spline.shape[0]):
                    pt1 = NED2IMG_single(fitted_spline[k-1, 0], fitted_spline[k-1, 1], self.__minX, self.__minY, self.__mppr)
                    pt2 = NED2IMG_single(fitted_spline[k, 0], fitted_spline[k, 1], self.__minX, self.__minY, self.__mppr)
                    cv.line(self.mapVis, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 200, thickness=1)

        ## visualize prior positions of the target 
        # for pt in self.object_points_pix:
        #     cv.circle(newMap, (pt[0], pt[1]), 2, 127, 2)
        #if not est_pos_ned is None:
        #    pt_next = NED2IMG_single(est_pos_ned[0], est_pos_ned[1], self.__minX, self.__minY, self.__mppr)
        #    cv.circle(newMap, (pt_next[0], pt_next[1]), 2, 255, 2)


    def updateOccupancyMap(self):
        current_frame = cv.imread(frame_path)
        normal_mat = self.midas_object.seeDepth(current_frame)
        h, w = normal_mat.shape
        DIST = np.zeros((h,w))

    def predict(self, imu_meas, cam_ps, object_pose, t, _dt, frame_path, score_map=None):
        self.updateOccupancyMap()
        # buffer last object poses
        if object_pose is None:
            obj_xy_ned = None
        else:
            pt = NED2IMG_single(object_pose[1], object_pose[2], self.__minX, self.__minY, self.__mppr)
            self.object_points_pix.append(pt)
            obj_xy_ned = [object_pose[1], object_pose[2]]

        # pose_est_ned, spline, _ = self.estimator.update(obj_xy_ned, t, self.const_dt) 
        prob_points, spline, _ = self.estimator.update(obj_xy_ned, t, _dt) 

        self.updateMap(prob_points, spline)
        pan_scan, cornersInMap = self.scan(imu_meas, cam_ps)
        picture = self.takePicture(cornersInMap)
        return self.getOptimalROI(picture)

    def takePicture(self, cornersInMap):
        if self.map is None:
            return None
        homographicTransform, _ = cv.findHomography(cornersInMap, self._imageCorners)
        pic = cv.warpPerspective(self.map, homographicTransform, self._pixelMapSize)
        if self.show_demo:
            cv.imshow("picture", pic)
        return pic
    
    # def getOptimalROI(self, score_map, pan_scan): # supposed to be the complete version
    # def getOptimalROI(self, score_map, picture): # advanced version
    def getOptimalROI(self, picture): # primary version
        ret,thresh = cv.threshold(picture, self.picturing_thresh, 255, 0)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            # find the biggest countour (c) by the area
            c = max(contours, key = cv.contourArea)
            x,y,w,h = cv.boundingRect(c)
            # self.lastOptimalROI = (x,y,w,h)
            ret = (x,y,w,h)
            cv.rectangle(thresh, (x,y), (x+w, y+h), 125, 2)
        else:
            ret = None

        # cv.imwrite('/home/hamid/ffs/img{}.jpg'.format(self.iter), thresh)
        cv.imshow('thresh', thresh)
        # cv.imwrite('/home/hamid/ffs/ig{}.jpg'.format(self.iter), picture)
        cv.imshow('picture', picture)
        self.iter += 1
        return ret
        # elif self.lastOptimalROI is None:
        #     raise Exception("No contour while even the fist optimal ROI is not set")
        # return self.lastOptimalROI


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
            if minX > dist_x_low:
                minX = dist_x_low
            if maxX < dist_x_high:
                maxX = dist_x_high
            if minY > dist_y_low:
                minY = dist_y_low
            if maxY < dist_y_high:
                maxY = dist_y_high
        assert (maxX > minX and maxY > minY), "min may not be more than max"
        self.__mapWidth_pix = int((maxX - minX) / self.__mppr)
        self.__mapHeight_pix = int((maxY - minY) / self.__mppr)
        self.__minX = minX
        self.__minY = minY
        # self.map = np.zeros((self.__mapWidth_pix, self.__mapHeight_pix))
        self.min_r_pix = int(self.min_r_meter / self.__mppr)
        self.max_r_pix = int(self.max_r_meter / self.__mppr)
    
