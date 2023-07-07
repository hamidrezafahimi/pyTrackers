from matplotlib import pyplot as plt
import numpy as np
import json
from pathlib import Path
data_path = str(Path(__file__).parent.resolve()) + "/../../results/"

target_poses_ned = np.loadtxt(data_path+'park_mavic_1_target_poses.txt', delimiter=',')
camera_poses_ned = np.loadtxt(data_path+'park_mavic_1_cam_poses.txt', delimiter=',')
viot_poses_ned = np.loadtxt(data_path+'MIXFORMER_park_mavic_1_viot_pose3ds.txt', delimiter=',')
tracker_poses_ned = np.loadtxt(data_path+'MIXFORMER_park_mavic_1_raw_pose3ds.txt', delimiter=',')

plt.plot(target_poses_ned[:,1], target_poses_ned[:,2], label='target gt')
plt.plot(camera_poses_ned[:,1], camera_poses_ned[:,2], label='camera gt')
plt.plot(viot_poses_ned[:,1], viot_poses_ned[:,2], label='viot estimation')
plt.plot(tracker_poses_ned[:,1], tracker_poses_ned[:,2], label='tracker estimation')
plt.scatter(camera_poses_ned[0,1], camera_poses_ned[0,2])
plt.scatter(target_poses_ned[0,1], target_poses_ned[0,2])
plt.scatter(viot_poses_ned[0,1], viot_poses_ned[0,2])
plt.scatter(tracker_poses_ned[0,1], tracker_poses_ned[0,2])
plt.legend()
plt.grid(True)
plt.show()