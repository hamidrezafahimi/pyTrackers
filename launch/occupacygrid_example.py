import cv2 as cv
import numpy as np
import os, sys
# from .panorama_scanner import panorama_scannera 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extensions.camera_kinematics import CameraKinematics
from extensions.midas.midas import Midas

def plotOccupancy(x_axis, y_axis, z_axis):
    voxel_size = 0.2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in zip(x_axis, y_axis, z_axis):
        ax.bar3d(x, y, z, voxel_size, voxel_size, voxel_size, shade=True, color='blue')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    plt.title("3D Voxel Plot")
    plt.show()


def plotPositions(x_axis, y_axis, z_axis):
    x_min, x_max = x_axis.min(), x_axis.max()
    y_min, y_max = y_axis.min(), y_axis.max()
    z_min, z_max = z_axis.min(), z_axis.max()

    x_lim = [x_min - 0.05 * np.abs(x_min), x_max + 0.05 * np.abs(x_max)]
    y_lim = [y_min - 0.05 * np.abs(y_min), y_max + 0.05 * np.abs(y_max)]
    z_lim = [z_min - 0.05 * np.abs(z_min), z_max + 0.05 * np.abs(z_max)]
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter(x_axis, y_axis, z_axis)

    bounding_box = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        ax.plot3D(*zip(bounding_box[edge[0]], bounding_box[edge[1]]), color='b')

    ax.set_title('%0*d' % (8, i) + ".png")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    plt.show()


# paths
data_path = str(Path(__file__).parent.resolve()) + "/../dataset/VIOT/park_mavic_1/"
camera_states = np.loadtxt(data_path+'camera_states.txt', delimiter=',')

# objects
kin = CameraKinematics(cx=220.0, cy=165.0, w=440, h=330, hfov=66.0, ref=camera_states[0,1:4])
midas_object = Midas("midas_v21_384.pt", "midas_v21_384")
data_root = "/home/ali/workspaces/as/extensions/midas/output/"



num_of_numbers = camera_states.shape[0]
for i in range(758, 759):
    z_axis = []
    x_axis = []
    y_axis = []
    img_path = data_root + '%0*d' % (8, i) + ".png"
    normal_mat = cv.imread(img_path)
    normal_mat = midas_object.reshape_matrix_by_percentage(normal_mat, 2, 2)
    h, w = normal_mat.shape
    DIST = np.zeros((h,w))
    for x in range(h):
        for y in range(w):
            r_max ,r_min = kin.point_to_r_max_min(y, x, camera_states[i,4:7], camera_states[i,1:4], 3)
            dist = ((normal_mat[x, y] - 0) / (255 - 0)) * (r_max -r_min) + r_min
            data = [r_min, dist, r_max]
            data = midas_object.getNormalized(0, 1, data)
            px2pose = kin.pixel_to_pose(y, x, camera_states[i,4:7], camera_states[i,1:4], data[1]*3)
            DIST[x,y] = (1-data[1])*3
            x_axis.append(px2pose[0])
            y_axis.append(px2pose[1])
            z_axis.append(px2pose[2])
            # print("Pixel h: " + str(x) + " w: " + str(y) + " Dist Value: " + str(DIST[x,y]) + " Midas Value: " + str(normal_mat[x, y]))

    # Convert lists to NumPy arrays
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    z_axis = np.array(z_axis)
    coords = np.vstack([np.vstack([x_axis, y_axis]), z_axis]).T
    x_min, x_max = x_axis.min(), x_axis.max()
    y_min, y_max = y_axis.min(), y_axis.max()
    z_min, z_max = z_axis.min(), z_axis.max()

    x_lim = [x_min - 0.05 * np.abs(x_min), x_max + 0.05 * np.abs(x_max)]
    y_lim = [y_min - 0.05 * np.abs(y_min), y_max + 0.05 * np.abs(y_max)]
    z_lim = [z_min - 0.05 * np.abs(z_min), z_max + 0.05 * np.abs(z_max)]

    # plotPositions(x_axis, y_axis, z_axis)
    # plotOccupancy(x_axis, y_axis, z_axis)
    min_mat = np.array([list(np.ones(coords[:,0].shape)*x_min), list(np.ones(coords[:,1].shape)*y_min), list(np.zeros(coords[:,1].shape))])
    norm_coords = coords - min_mat.T
    mppr = 1
    height_pix = int((x_max - x_min)/mppr)+10
    width_pix = int((y_max - y_min)/mppr)+10
    print('----')
    print(height_pix, width_pix)
    new_map = np.zeros((height_pix, width_pix))
    print(x_min, y_min, z_min)
    print(x_max, y_max, z_max)
    print(norm_coords.shape)
    for idx in range(len(x_axis)):
        x_idx = int((x_axis[idx]-x_min)/mppr)
        y_idx = int((y_axis[idx]-y_min)/mppr)
        print(x_axis[idx], y_axis[idx], x_idx, y_idx)
        new_map[y_idx, x_idx] = z_axis[idx]
    ret,thresh = cv.threshold(new_map, 1.5, 255, 0)
    frame = cv.resize(thresh, (100,100))
    cv.imshow("window_name", frame)
    cv.waitKey()
    print (DIST.shape)





    