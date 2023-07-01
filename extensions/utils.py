import numpy as np
import utm
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy import ndimage


def gps_to_ned(ref_loc, loc):
    y_ref, x_ref, _, _ = utm.from_latlon(ref_loc[0], ref_loc[1])
    pose_ref_utm = np.array( [x_ref, y_ref, -ref_loc[2]] )

    y, x, _, _ = utm.from_latlon(loc[0], loc[1])
    pose_utm = [x,y,-loc[2]]
    pose_ned = pose_utm-pose_ref_utm

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

def toSpherecalCoords(vec):

    if vec is None:
        return None

    x = vec[0]
    y = vec[1]
    z = vec[2]

    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arccos( z / r )

    if x>0:
        phi = np.arctan(y/x)
    elif x<0 and y>=0:
        phi = np.arctan(y/x) + np.pi
    elif x<0 and y<0:
        phi = np.arctan(y/x) - np.pi
    elif x==0 and y>0:
        phi = np.pi
    elif x==0 and y<0:
        phi = -np.pi

    return np.array([r,th, phi])

def toCartesianCoords(vec):

    if vec is None:
        return None

    r = vec[0]
    th = vec[1]
    phi = vec[2]

    x = r*np.cos(phi)*np.sin(th)
    y = r*np.sin(phi)*np.sin(th)
    z = r*np.cos(th)

    return np.array([x, y, z])

def angleDifference(ang1, ang2):
    PI  = np.pi

    a = ang1 - ang2
    if a > PI:
        a -= 2*PI
    if a < -PI:
        a += 2*PI

    return a