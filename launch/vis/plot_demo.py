from utility import draw_pyramid, get_euler_angles, set_axes_equal
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
data_path = str(Path(__file__).parent.resolve()) + "/../../results/"


if __name__ == '__main__':

    target_poses_ned = np.loadtxt(data_path+'park_mavic_1_target_poses.txt', delimiter=',')
    camera_poses_ned = np.loadtxt(data_path+'park_mavic_1_cam_poses.txt', delimiter=',')

    num_of_numbers = target_poses_ned.shape[0]

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(elev=-160, azim=-120)
    plt.grid(True)

    for i in range(1, num_of_numbers):
        plt.cla()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        los = np.array([[target_poses_ned[i, 1]], [target_poses_ned[i, 2]], 
                        [target_poses_ned[i, 3]]]) - np.array([[camera_poses_ned[i, 1]], 
                        [camera_poses_ned[i, 2]], [camera_poses_ned[i, 3]]])
        ax.plot3D([target_poses_ned[i, 1], camera_poses_ned[i, 1]],
                  [target_poses_ned[i, 2], camera_poses_ned[i, 2]],
                  [target_poses_ned[i, 3], camera_poses_ned[i, 3]], '--')
        if i > 50:
            oldIdx = 50
        else:
            oldIdx = i
        ax.plot3D(target_poses_ned[i-oldIdx:i, 1], target_poses_ned[i-oldIdx:i, 2], 
                  target_poses_ned[i-oldIdx:i, 3], 
                color = 'blue')
        ax.plot3D(camera_poses_ned[i-oldIdx:i, 1], camera_poses_ned[i-oldIdx:i, 2], 
                  camera_poses_ned[i-oldIdx:i, 3], 
                color = 'red')
        draw_pyramid(ax, 4, 10, 10, camera_poses_ned[i, 1:4], camera_poses_ned[i, 4], 
                     -camera_poses_ned[i, 5], camera_poses_ned[i, 6], 'blue')
        a1, a2, a3 = get_euler_angles(np.array([0, 0, 1]), 
                                      np.array([los[0,0], los[1,0], los[2,0]]))
        draw_pyramid(ax, 4, 10, 10, camera_poses_ned[i, 1:4], a1, a2, a3, 'red')
        set_axes_equal(ax)
        plt.pause(0.01)
