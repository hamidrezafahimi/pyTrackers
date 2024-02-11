import numpy as np
import utm
import time

def all_to_ned(data):
    ref_loc = data[0,1:4]
    out = []
    for k in range(data.shape[0]):
        out.append([data[k,0], *gps_to_ned(ref_loc, data[k,1:4]), *data[k,4:]])
    return np.array(out)

def all_to_utm(data):
    out = data
    for k in range(data.shape[0]):
        out[k, 2], out[k, 1], _, _ = utm.from_latlon(data[k, 1], data[k, 2])
    return out

def utm_to_ned(ref_loc, loc):
    pose_ref_utm = np.array( [ref_loc[0], ref_loc[1], -ref_loc[2]] )
    pose_utm = [loc[0],loc[1],-loc[2]]
    pose_ned = pose_utm-pose_ref_utm
    return np.array(pose_ned)

def latLon_to_utm(loc):
    y, x, _, _ = utm.from_latlon(loc[0], loc[1])
    return [x, y]

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