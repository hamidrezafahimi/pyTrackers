import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math as m

def rotate_point(point, phi, theta, psi):
	""" Rotates a given point based on euler angles
	It is equivalent to rotating the old C.S. in which the point lies and get coordinates of the
	point in the new C.S.
	The order of rotation is: 
		rotate by phi (around x) -> rotate by theta (around y) -> rotate by psi (around z)
	Thus, the order of matrix multiplication becomes: Rz(psi) * Ry(theta) * Rx(phi)
	"""
	assert(point.shape == (3,1) or point.shape == (1,3))
	if (point.shape == (1,3)): point = point.T
	if (point.shape == (3,)): 
		point = np.array([[point[0]], [point[1]], [point[2]]])
	def Rx(alpha):
		return np.matrix([[ 1, 0           , 0           ],
						[ 0, m.cos(alpha),-m.sin(alpha)],
						[ 0, m.sin(alpha), m.cos(alpha)]])
	def Ry(alpha):
		return np.matrix([[ m.cos(alpha), 0, m.sin(alpha)],
						[ 0           , 1, 0           ],
						[-m.sin(alpha), 0, m.cos(alpha)]])
	def Rz(alpha):
		return np.matrix([[ m.cos(alpha), -m.sin(alpha), 0 ],
						[ m.sin(alpha), m.cos(alpha) , 0 ],
						[ 0           , 0            , 1 ]])
	R = Rz(psi) * Ry(theta) * Rx(phi)
	return np.array(R * point)

def transform_point(point, phi, theta, psi, T):
	""" Performs a rotation as well as moving the C.S. origin to a given point T
	"""
	assert(T.shape == (3,1) or T.shape == (1,3) or T.shape == (3,))
	if (T.shape == (1,3)): T = T.T
	if (T.shape == (3,)): 
		T = np.array([[T[0]], [T[1]], [T[2]]])
	return np.array(rotate_point(point, phi, theta, psi) + T)

def draw_pyramid(ax, height, hfov_deg, vfov_deg, pose, yaw, pitch, roll, facecolors='cyan'):
	wx = 2 * height * np.tan((hfov_deg * np.pi / 180) / 2)
	wy = 2 * height * np.tan((vfov_deg * np.pi / 180) / 2)
	cpose = transform_point(np.array([[0], [0], [0]]), yaw, pitch, roll, pose)
	c1 = transform_point(np.array([[-wx], [-wy], [height]]), yaw, pitch, roll, pose)
	c2 = transform_point(np.array([[wx], [-wy], [height]]), yaw, pitch, roll, pose)
	c3 = transform_point(np.array([[wx], [wy], [height]]), yaw, pitch, roll, pose)
	c4 = transform_point(np.array([[-wx], [wy], [height]]), yaw, pitch, roll, pose)
    # vertices of a pyramid
	v = np.array([list(*c1.T), list(*c2.T), list(*c3.T), list(*c4.T), list(*cpose.T)])
	# ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
    # generate list of sides' polygons of our pyramid
	verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
    	[v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]
    # plot sides
	ax.add_collection3d(Poly3DCollection(verts, 
        facecolors=facecolors, linewidths=1, edgecolors='r', alpha=.25))

def get_euler_angles(v1, v2):
    # Find axis of rotation
    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)

    # Find angle of rotation
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)

    # Create rotation matrix
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x = axis[0]
    y = axis[1]
    z = axis[2]
    rot_matrix = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                           [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                           [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])

    # Extract Euler angles
    sy = np.sqrt(rot_matrix[0,0] * rot_matrix[0,0] + rot_matrix[1,0] * rot_matrix[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rot_matrix[2,1], rot_matrix[2,2])
        y = np.arctan2(-rot_matrix[2,0], sy)
        z = np.arctan2(rot_matrix[1,0], rot_matrix[0,0])
    else:
        x = np.arctan2(-rot_matrix[1,2], rot_matrix[1,1])
        y = np.arctan2(-rot_matrix[2,0], sy)
        z = 0

    return x, y, z

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
