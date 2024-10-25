import cv2 as cv
import numpy as np
import os, sys
# from .panorama_scanner import panorama_scannera 
import matplotlib.pyplot as plt
from pathlib import Path
import copy

def get_ned_wrt_ref(ref_loc, loc):
    # y_ref, x_ref, _, _ = utm.from_latlon(ref_loc[0], ref_loc[1])
    # pose_ref_utm = np.array( [x_ref, y_ref, -ref_loc[2]] )

    # y, x, _, _ = utm.from_latlon(loc[0], loc[1])
    # pose_utm = [x,y,-loc[2]]
    pose_ned = loc-ref_loc
    return np.array(pose_ned)

def make_DCM(eul):

    phi = eul[0]
    theta = eul[1]
    psi = eul[2]

    DCM = np.zeros((3,3))
    DCM[0,0] = np.cos(psi)*np.cos(theta)
    DCM[0,1] = np.sin(psi)*np.cos(theta)
    DCM[0,2] = -np.sin(theta)
    DCM[1,0] = np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi)
    DCM[1,1] = np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi)
    DCM[1,2] = np.cos(theta)*np.sin(phi)
    DCM[2,0] = np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi)
    DCM[2,1] = np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)
    DCM[2,2] = np.cos(theta)*np.cos(phi)

    return DCM

class CameraKinematics:

    def __init__(self, ref, cx, cy, f=None, w=None, h=None, hfov=None):
        self.ref_loc = ref
        self.ref_loc[2] = 0
        self._cx = cx
        self._cy = cy
        self._hfov = hfov
        self._w = w
        self._h = h
        if f is not None:
            self._f = f
        elif f is None and (hfov is not None and w is not None and h is not None):
            self._f = (0.5 * w * (1.0 / np.tan((hfov/2.0)*np.pi/180)))
        else:
            raise ValueError('At least one of arguments "f" or "hfov" must have value.')

    def body_to_inertia(self, body_vec, eul):
        if body_vec is None:
            return None
        ## calculate a DCM and find transpose that takes body to inertial
        DCM_ib = make_DCM(eul).T
        ## return vector in inertial coordinates
        return np.matmul(DCM_ib, body_vec)

    def inertia_to_body(self, in_vec, eul):
        ## calculate a "DCM" using euler angles of camera body, to convert vector
        ## from inertial to body coordinates
        DCM_bi = make_DCM(eul)
        ## return the vector in body coordinates
        return np.matmul(DCM_bi, in_vec)


    def cam_to_body(self, rect, wrt_leg=False):
        if rect is None:
            return None
        ## converting 2d rectangle to a 3d vector in camera coordinates
        vec = self.to_direction_vector(rect, self._cx, self._cy, self._f, wrt_leg)
        ## for MAVIC Mini camera, the body axis can be converted to camera
        ## axis by a 90 deg yaw and a 90 deg roll consecutively. then we transpose
        ## it to get camera to body
        DCM_bc = make_DCM([90*np.pi/180, 0, 90*np.pi/180]).T
        return np.matmul(DCM_bc, vec)

    def body_to_cam(self, vec):
        ## for MAVIC Mini camera, the body axis can be converted to camera
        ## axis by a 90 deg yaw and a 90 deg roll consecutively.
        DCM_cb = make_DCM([90*np.pi/180, 0, 90*np.pi/180])
        return np.matmul(DCM_cb, vec)

    def to_direction_vector(self, rect, cx, cy, f, wrt_leg=False):
        ## find center point of target
        if wrt_leg:
            ## In this case, the direction vector is calculated wrt middle of bottom side of rect
            center = np.array([rect[0]+rect[2]/2, rect[1]+rect[3]])
        else:
            center = np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2])
        ## project 2d point from image plane to 3d space using a simple pinhole
        ## camera model
        w = np.array( [ (center[0] - cx) , (center[1] - cy), f] )
        return w/np.linalg.norm(w)

    def from_direction_vector(self, dir, cx, cy, f):
        ## avoid division by zero
        if dir[2] < 0.01:
            dir[2] = 0.01
        ## calculate reprojection of direction vectors to image plane using a
        ## simple pinhole camera model
        X = cx + (dir[0] / dir[2]) * f
        Y = cy + (dir[1] / dir[2]) * f
        return (int(X),int(Y))

    def scale_vector(self, v, z):
        if v is None:
            return None
        ## scale a unit vector v based on the fact that third component should be
        ## equal to z
        max_dist = 50
        if v[2] > 0:
            factor = np.abs(z) / np.abs(v[2])
            if np.linalg.norm(factor*v) < max_dist:
                return factor*v
            else:
                return max_dist*v
        elif v[2] <= 0:
            return max_dist*v

    def limit_vector_to_fov(self, vector):
        ## angle between target direction vector and camera forward axis
        angle = np.arccos( np.dot(vector, np.array([0,0,1])) / np.linalg.norm(vector) )
        ## rotation axis which is perpendicular to both target vector and camera 
        ## axis
        axis = np.cross( np.array([0,0,1]), vector  / np.linalg.norm(vector) )
        last_rotated_vec = None
        for i in range(90):
            rotation_degrees = i * np.sign(angle)
            rotation_radians = np.radians(rotation_degrees)
            rotation_axis = axis / np.linalg.norm(axis)
            rotation_vector = rotation_radians * rotation_axis
            rotation = R.from_rotvec(rotation_vector)
            rotated_vec = rotation.apply( np.array([0,0,1]) )
            ## reproject to image plane
            reproj_vec = self.from_direction_vector(rotated_vec, self._cx, self._cy, self._f)
            if reproj_vec[0] >= self._w or reproj_vec[0] <= 0 or \
               reproj_vec[1] >= self._h or reproj_vec[1] <= 0:
                break
            last_rotated_vec = rotated_vec
        reproj = self.from_direction_vector(vector, self._cx, self._cy, self._f)
        if reproj[0] >= self._w or reproj[0] <= 0 or \
           reproj[1] >= self._h or reproj[1] <= 0:
            # print("out ", reproj_vec)
            return last_rotated_vec
        else:
            # print("in ", reproj)
            return vector

    def get_camera_frame_vecs(self, eul):
        ## convert image corners from a point in "image coordinates" to a vector
        ## in "camera body coordinates"
        top_left = self.cam_to_body([-1,-1,2,2])
        top_right = self.cam_to_body([self._w-1,-1,2,2])
        bottom_left = self.cam_to_body([-1,self._h-1,2,2])
        bottom_right = self.cam_to_body([self._w-1,self._h-1,2,2])
        ## convert image corners from a vector in "camera body coordinates" to
        ## a vector in "inertial coordinates"
        top_left_inertia_dir = self.body_to_inertia(top_left, eul)
        top_right_inertia_dir = self.body_to_inertia(top_right, eul)
        bottom_left_inertia_dir = self.body_to_inertia(bottom_left, eul)
        bottom_right_inertia_dir = self.body_to_inertia(bottom_right, eul)
        return (top_left_inertia_dir,top_right_inertia_dir,\
                bottom_left_inertia_dir,bottom_right_inertia_dir)
        
    def rect_to_pose(self, rect, imu_meas, cam_ps, wrt_leg=False):
        ## convert gps lat, lon positions to a local cartesian coordinate
        cam_pos = get_ned_wrt_ref(self.ref_loc, cam_ps)
        if rect is None:
            return None, cam_pos
        ## convert target from a rect in "image coordinates" to a vector
        ## in "camera body coordinates"
        body_dir = self.cam_to_body(rect, wrt_leg)
        ## convert target from a vector in "camera body coordinates" to a vector
        ## in "inertial coordinates"
        inertia_dir = self.body_to_inertia(body_dir, imu_meas)
        ## calculate target pos
        target_pos = self.scale_vector(inertia_dir, cam_pos[2]) + cam_pos
        return target_pos, cam_pos

    def get_camera_fov_area(self, imu_meas, cam_ps):
        top_left_inertia_dir,top_right_inertia_dir,\
            bottom_left_inertia_dir,bottom_right_inertia_dir = \
            self.get_camera_frame_vecs(imu_meas)
        cam_pos = self.get_cam_pos_ned(cam_ps)
        pos1 = self.scale_vector(top_left_inertia_dir, cam_ps[2]) + cam_pos
        pos2 = self.scale_vector(top_right_inertia_dir, cam_ps[2]) + cam_pos
        pos3 = self.scale_vector(bottom_right_inertia_dir, cam_ps[2]) + cam_pos
        pos4 = self.scale_vector(bottom_left_inertia_dir, cam_ps[2]) + cam_pos
        return pos1, pos2, pos3, pos4
    
    def point_to_pose(self, pix_x, pix_y, imu_meas, cam_ps):
        cam_pos = self.get_cam_pos_ned(cam_ps)
        b_vec = self.cam_to_body([pix_x-1,pix_y-1,2,2])
        # print(b_vec)
        inertia_dir = self.body_to_inertia(b_vec, imu_meas)
        return self.scale_vector(inertia_dir, cam_ps[2]) + cam_pos

    def point_to_r_max_min(self, pix_x, pix_y, imu_meas, cam_ps, roof_height):
        cam_z = self.get_cam_pos_ned(cam_ps)[2]
        b_vec = self.cam_to_body([pix_x-1,pix_y-1,2,2])
        inertia_dir = self.body_to_inertia(b_vec, imu_meas)
        r_min = np.linalg.norm(self.scale_vector(inertia_dir, cam_ps[2]+roof_height))
        r_max = np.linalg.norm(self.scale_vector(inertia_dir, cam_ps[2]))
        return r_max, r_min

    def get_cam_pos_ned(self, cam_ps):
        return get_ned_wrt_ref(self.ref_loc, cam_ps)

    def pose_to_limited_rect(self, pose, cam_pos, imu_meas, rect_sample):
        if pose is None or pose[0] is None:
            return None
        inertia_dir = pose - cam_pos
        if np.linalg.norm(inertia_dir) != 0:
            inertia_dir = inertia_dir / np.linalg.norm(inertia_dir)
            ## convert new estimate of target direction vector to body coordinates
            body_dir_est = self.inertia_to_body( inertia_dir, imu_meas)
            ## convert body to cam coordinates
            cam_dir_est = self.body_to_cam(body_dir_est)
            cam_dir_est = self.limit_vector_to_fov(cam_dir_est)
            ## reproject to image plane
            center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)
            return (int(center_est[0]-rect_sample[2]/2), \
                    int(center_est[1]-rect_sample[3]/2), rect_sample[2], rect_sample[3]) # rect_est
        else:
            return None


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'extensions')))
# from midas.midas import Midas
data_root = "/home/hamid/w/REPOS/pyTrackers/extensions/midas/output/"
data_path = str(Path(__file__).parent.resolve()) + "/../dataset/VIOT/park_mavic_1/"
camera_states = np.loadtxt(data_path+'camera_states.txt', delimiter=',')
num_of_numbers = camera_states.shape[0]

print(len(camera_states[0]))
kin = CameraKinematics(copy.deepcopy(camera_states[0, 1:4]),
                    cx=220.0, cy=165.0, 
                    w=440,
                    h=330, hfov=66.0)

for i in range(2, num_of_numbers):
    # print(camera_states[i, 1])
    img_path = data_root + '%0*d' % (8, i) + ".png"
    normal_mat = cv.imread(img_path)
    h, w, _ = normal_mat.shape
    DIST = np.zeros((h,w))

    # print(self.point_to_pose(0,0, imu_meas, cam_ps))
    
    for i in range(h):
        for j in range(w):
            r_min ,r_max = kin.point_to_r_max_min(j, i, camera_states[i,4:7], camera_states[i,1:4], 3)
            print(normal_mat[i, j])
            print(r_min)
            # dist = (normal_mat[i, j] * (r_max -r_min) / 255) + r_min
            old_min, old_max = 0, 255
            new_min, new_max = r_max, r_min 
            # dist = ((normal_mat[i, j] - 0) / (255 - 0)) * (r_max -r_min) + r_min
            dist = new_min + (normal_mat[i, j][0] - old_min) * (new_max - new_min) / (old_max - old_min)
            print(dist)
            DIST[i,j] = dist
            # print ("Pixel h: " + str(i) + " w:" + str(j) + " Dist Value: " + str(dist) + " Midas Value: " + str(normal_mat[i, j]))
        

    cv.imshow("normal_mat", DIST)
    cv.waitKey(1)
    