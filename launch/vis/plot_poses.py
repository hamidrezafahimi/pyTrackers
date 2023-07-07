from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
data_path = str(Path(__file__).parent.resolve()) + "/../../results/"

target_poses_ned = np.loadtxt(data_path+'park_mavic_1_target_poses.txt', delimiter=',')
camera_poses_ned = np.loadtxt(data_path+'park_mavic_1_cam_poses.txt', delimiter=',')

plt.plot(target_poses_ned[:,1], target_poses_ned[:,2], label='t')
plt.plot(camera_poses_ned[:,1], camera_poses_ned[:,2], label='c')
plt.scatter(camera_poses_ned[0,1], camera_poses_ned[0,2])
plt.scatter(target_poses_ned[0,1], target_poses_ned[0,2])
plt.legend()
plt.grid(True)
plt.show()