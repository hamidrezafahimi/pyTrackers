import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 

ax = plt.figure().add_subplot(111, projection='3d')

cpose = [0, 0, 0]
height = 1
hfov = 60
vfov = 60

wx = 2 * height * np.tan((hfov * np.pi / 180) / 2)
wy = 2 * height * np.tan((vfov * np.pi / 180) / 2)

c1 = np.array([-wx, -wy, height])
c2 = np.array([wx, -wy, height])
c3 = np.array([wx, wy, height])
c4 = np.array([-wx, wy, height])

# vertices of a pyramid
v = np.array([c1, c2, c3, c4, cpose])
ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

# generate list of sides' polygons of our pyramid
verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
 [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]

# plot sides
ax.add_collection3d(Poly3DCollection(verts, 
 facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

plt.show()