import numpy as np
import cv2 as cv
from lib.utils.plotting import plot_kinematics
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_path = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(0, root_path)
from lib.utils import gps_to_ned, toSpherecalCoords, toCartesianCoords, angleDifference
from .camera_kinematics import CameraKinematics

class VIOTrack(CameraKinematics):
    def __init__(self, factor, cx, cy, ref, f=None, w=None, h=None, hfov=None, vis=True):
        super().__init__(ref, cx, cy, f, w, h, hfov)
        
        self._init = False
        self._diff = np.array( [0,0,0] )
        self._inertia_dir_before = np.array( [0,0,0] )
        self._inertia_dir_after = np.array( [0,0,0] )
        self._last_rect = (0,0,0,0)
        self._interp_factor = factor
        self._diff_buff = []
        self._pos_buff = []
        self._pos_est = None
        self._pos_buff_size = 40
        self._last_target_states = [False]
        self._vis=vis
        if vis:
            self._fig_3d=plt.figure(0)
            self._ax_3d=plt.axes(projection ='3d')
            self._ax_3d.set_title('Kinematics Plot')

    def updateRect3D(self, states, image, rect=None):

        imu_meas = states[4:7]
        gps_meas = states[1:4]
        target_pos, cam_pos = self.rect_to_pose(rect, imu_meas, gps_meas)
        
        if rect is not None:
            self._last_rect = rect
            ## if target is just found, empty the observation buffer to prevent
            ## oscilations around target
            if len(self._last_target_states) >= 5:
                if np.sum( np.array(self._last_target_states[2:5]) ) >= 2  and \
                   np.sum( np.array(self._last_target_states[0:3]) ) <= 1:
                    self._pos_buff = []

            ## buffer target positions
            if len(self._pos_buff) > self._pos_buff_size:
                del self._pos_buff[0]
            self._pos_buff.append([states[0], target_pos[0], target_pos[1], target_pos[2]])

        ## if target just disappeared, eliminate some of the last buffered observations,
        ## because target's box is having misleading shakes before being lost
        if rect is None and self._last_target_states[-1]:
            for i in range( int(0.2*len(self._pos_buff)) ):
                del self._pos_buff[-1]

        ## record last target states
        if len(self._last_target_states) >= 5:
            del self._last_target_states[0]
        self._last_target_states.append(not rect is None)
        
        ## calculate velocities for each consecutive pair of buffered target positions (used to 
        ## estimate a current velocity for the target)
        vs = []
        for i in range(1, len(self._pos_buff)):

            t0 = self._pos_buff[i-1][0]
            pos0 = self._pos_buff[i-1][1:4]

            t = self._pos_buff[i][0]
            pos = self._pos_buff[i][1:4]

            dx = np.array(pos) - np.array(pos0)
            dt = t-t0

            ## `simultaneous` or `far away` buffered positions are to be ignored in calculating 
            ## mean velocity
            if dt < 1 and dt!=0:
                vs.append(dx/dt)

        ## visualize buffered target positions as projected points in given image
        for data in self._pos_buff:
            pos = data[1:4]
            inertia_dir = pos - cam_pos
            if np.linalg.norm(inertia_dir) == 0:
                continue

            inertia_dir = inertia_dir / np.linalg.norm(inertia_dir)

            ## convert new estimate of target direction vector to body coordinates
            body_dir_est = self.inertia_to_body( inertia_dir, imu_meas)

            ## convert body to cam coordinates
            cam_dir_est = self.body_to_cam(body_dir_est)

            ## reproject to image plane
            center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)

            image = cv.circle(image, center_est, 2, (0, 255, 255), -1)

        ## calculate a mean velocity from velocity buffer and linearly estimate a new position 
        ## for target
        if len(vs)>0:
            v = np.mean(vs,0)
            dt = states[0] - self._pos_buff[-2][0]
            self._pos_est = self._pos_buff[-1][1:4] + self._interp_factor*v*dt

        if self._pos_est is None:
            return self._last_rect, [np.inf, np.inf, np.inf]

        ## based on the estimated position for the target (if there is any), reproject and get an
        ## estimated center for search area within image
        inertia_dir = self._pos_est - cam_pos
        if np.linalg.norm(inertia_dir) != 0:

            inertia_dir = inertia_dir / np.linalg.norm(inertia_dir)

            ## convert new estimate of target direction vector to body coordinates
            body_dir_est = self.inertia_to_body( inertia_dir, imu_meas)

            ## convert body to cam coordinates
            cam_dir_est = self.body_to_cam(body_dir_est)

            cam_dir_est = self.limit_vector_to_fov(cam_dir_est)

            ## reproject to image plane
            center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)

        ## estimated rectangle is obtained based on estimated center point for target's next 
        ## position, or center point corresponding to last buffered position for target
        rect_est = (int(center_est[0]-self._last_rect[2]/2), \
                    int(center_est[1]-self._last_rect[3]/2),
                    self._last_rect[2], self._last_rect[3])
        # image = cv.putText(image, '{:d}, {:d}, {:d}'.format(center_est[0], center_est[1],
        # len(vs)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv.LINE_AA)
        return rect_est, self._pos_est


    def updateRectSphere(self, imu_meas, rect=None):

        if rect is not None:
            self._last_rect = rect

        ## convert target from a rect in "image coordinates" to a vector
        ## in "camera body coordinates"
        body_dir = self.cam_to_body(rect)

        ## convert target from a vector in "camera body coordinates" to a vector
        ## in "inertial coordinates"
        inertia_dir = self.body_to_inertia(body_dir, imu_meas)

        ## represent inertia_dir in spherecal coordinates
        inertia_dir_sp = toSpherecalCoords(inertia_dir)

        if self._init:
            diff=np.array([0.0, 0.0, 0.0])
            if inertia_dir is not None:
                ## find the difference between new observation (inertia_dir) and last
                ## known direction (self._inertia_dir_before)
                diff = np.array([0.0, angleDifference(inertia_dir_sp[1], self._inertia_dir_before[1]), \
                                      angleDifference(inertia_dir_sp[2], self._inertia_dir_before[2])])


            ## if target is just found, empty the observation buffer to prevent
            ## oscilations around target
            if inertia_dir is not None and all(~np.array(self._last_target_states)):
                self._diff_buff = []

            ## make the differences smooth overtime by a moving average. this adds a dynamic to target
            ## direction vector.
            if len(self._diff_buff) > -self._interp_factor:
                del self._diff_buff[0]
                self._diff_buff.append(diff)
            else:
                self._diff_buff.append(diff)

            ## if target just disappeared, eliminate some of the last buffered observations,
            ## because target's box is having misleading shakes before being lost
            if inertia_dir is None and self._last_target_states[-1]:
                for i in range( int(0.4*len(self._diff_buff)) ):
                    del self._diff_buff[-1]

            ## record last target states
            if inertia_dir is None:
                if len(self._last_target_states) < 3:
                    self._last_target_states.append(False)
                else:
                    del self._last_target_states[0]
                    self._last_target_states.append(False)
            else:
                if len(self._last_target_states) < 3:
                    self._last_target_states.append(True)
                else:
                    del self._last_target_states[0]
                    self._last_target_states.append(True)

            self._diff = np.mean(self._diff_buff, 0)

            ## calculate new estimate for target's direction vector
            self._inertia_dir_after = self._inertia_dir_before + self._diff

            ## save this new estimate as last known direction in the memory
            self._inertia_dir_before = self._inertia_dir_after.copy()

        else:

            if inertia_dir is not None:
                ## initialize with first observation
                self._inertia_dir_before = inertia_dir_sp.copy()
                self._inertia_dir_after = inertia_dir_sp.copy()
                self._diff = np.array([0.0,0.0,0.0])

                self._init = True
            else:
                return None

        ## convert back to cartesian coordinates
        inertia_dir_after_ca = toCartesianCoords(self._inertia_dir_after)

        if self._vis:
            ## expressing camera frame by for vectors of its image corners in inertial
            ## frame
            corners = self.get_camera_frame_vecs(imu_meas,self._w,self._h)
            plot_kinematics(imu_meas, inertia_dir_after_ca, self._ax_3d, corners)

        ## convert new estimate of target direction vector to body coordinates
        body_dir_est = self.inertia_to_body( inertia_dir_after_ca, imu_meas)

        ## convert body to cam coordinates
        cam_dir_est = self.body_to_cam(body_dir_est)

        ## reproject to image plane
        center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)

        ## estimated rectangle
        rect_est = (int(center_est[0]-self._last_rect[2]/2), \
                    int(center_est[1]-self._last_rect[3]/2),
                    self._last_rect[2], self._last_rect[3])

        return rect_est


    def updateRect(self, imu_meas, rect=None):

        if rect is not None:
            self._last_rect = rect

        ## convert target from a rect in "image coordinates" to a vector
        ## in "camera body coordinates"
        body_dir = self.cam_to_body(rect)

        ## convert target from a vector in "camera body coordinates" to a vector
        ## in "inertial coordinates"
        inertia_dir = self.body_to_inertia(body_dir, imu_meas)

        if self._init:
            ## update difference vector only in case of new observation
            ## otherwise continue changing direction vector with last know
            ## speed
            diff=np.array([0.0,0.0,0.0])
            if inertia_dir is not None:
                ## find the difference between new observation (inertia_dir) and last
                ## known direction (self._inertia_dir_before)
                diff = inertia_dir - self._inertia_dir_before

                ## make the differences smooth overtime. this add a dynamic to target
                ## direction vector.
                self._diff = self._interp_factor*self._diff + (1-self._interp_factor)*diff

            ## calculate new estimate for target's direction vector
            self._inertia_dir_after = self._inertia_dir_before + self._diff

            ## ensure direction vector always has a length of 1
            self._inertia_dir_after = self._inertia_dir_after/np.linalg.norm(self._inertia_dir_after)

            ## save this new estimate as last known direction in the memory
            self._inertia_dir_before = self._inertia_dir_after.copy()

        else:

            if inertia_dir is not None:
                ## initialize with first observation
                self._inertia_dir_before = inertia_dir
                self._inertia_dir_after = inertia_dir
                self._diff = np.array([0.0,0.0,0.0])

                self._init = True
            else:
                return None

        if self._vis:
            ## expressing camera frame by for vectors of its image corners in inertial
            ## frame
            corners = self.get_camera_frame_vecs(imu_meas,self._w,self._h)
            plot_kinematics(imu_meas,self._inertia_dir_after,self._ax_3d,corners)

        ## convert new estimate of target direction vector to body coordinates
        body_dir_est = self.inertia_to_body(self._inertia_dir_after,imu_meas)

        ## convert body to cam coordinates
        cam_dir_est = self.body_to_cam(body_dir_est)

        ## reproject to image plane
        center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)

        ## estimated rectangle
        rect_est = (int(center_est[0]-self._last_rect[2]/2), \
                    int(center_est[1]-self._last_rect[3]/2),
                    self._last_rect[2], self._last_rect[3])

        return rect_est




