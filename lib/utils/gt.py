import numpy as np
from .geom import all_to_utm, latLon_to_utm, utm_to_ned


class FlatGroundTargetLocator:
    def __init__(self, t_data_path, r_loc):
        t_data = np.loadtxt(t_data_path, delimiter=',')
        self.target_data_utm = all_to_utm(t_data)
        self.ref_loc_utm = [*latLon_to_utm(r_loc), r_loc[2]]
        
    def calc_target_ned_poses(self, camera_data):
        target_ned_poses = []
        for k in range(camera_data.shape[0]):
            target_ned_poses.append([camera_data[k,0],
                                     *self.calc_target_ned_pose(camera_data[k,0])])
        return np.array(target_ned_poses)

    def calc_target_ned_pose(self, t):
        tgt_utm_pose = [*self.interpTargetData(t), 0]
        return utm_to_ned(self.ref_loc_utm, tgt_utm_pose) #tgt_ned_pose
    
    def interpTargetData(self, t):
        out_x = np.interp(t, self.target_data_utm[:,0], self.target_data_utm[:,1])
        out_y = np.interp(t, self.target_data_utm[:,0], self.target_data_utm[:,2])
        return [out_x, out_y]

def calc_pose_error(gt_poses, poses):
    errors = []
    for k in range(1, gt_poses.shape[0]):
        errors.append([k, abs(np.sqrt((gt_poses[k,1]-poses[k-1,1])**2 + 
                                  (gt_poses[k,2]-poses[k-1,2])**2))])
    return np.array(errors)