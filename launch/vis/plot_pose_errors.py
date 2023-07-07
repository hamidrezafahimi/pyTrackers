from matplotlib import pyplot as plt
import numpy as np
import sys
from pathlib import Path
data_path = str(Path(__file__).parent.resolve()) + "/../../results/"
sys.path.insert(0, str(Path(__file__).parent.resolve()) + "/../..")
from lib.utils import calc_pose_error

target_poses_ned = np.loadtxt(data_path+'park_mavic_1_target_poses.txt', delimiter=',')
viot_poses_ned = np.loadtxt(data_path+'MIXFORMER_park_mavic_1_viot_pose3ds.txt', delimiter=',')
tracker_poses_ned = np.loadtxt(data_path+'MIXFORMER_park_mavic_1_raw_pose3ds.txt', delimiter=',')

viot_pose_error = calc_pose_error(target_poses_ned, viot_poses_ned)
tracker_pose_error = calc_pose_error(target_poses_ned, tracker_poses_ned)

plt.plot(viot_pose_error[:,0], viot_pose_error[:,1], label='viot estimation error (m)')
plt.plot(tracker_pose_error[:,0], tracker_pose_error[:,1], label='tracker estimation error (m)')

plt.legend()
plt.grid(True)
plt.show()