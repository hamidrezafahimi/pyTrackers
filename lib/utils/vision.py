import numpy as np
import cv2

def to_color_map(score,sz):
    score = cv2.resize(score, sz)
    score -= score.min()
    score = score / score.max()
    score = (score * 255).astype(np.uint8)
    # score = 255 - score
    score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
    return score

def calAUC(value_list):
    length=len(value_list)
    delta=1./(length-1)
    area=0.
    for i in range(1,length):
        area+=(delta*((value_list[i]+value_list[i-1])/2))
    return area


def cos_window(sz):
    """
    width, height = sz
    j = np.arange(0, width)
    i = np.arange(0, height)
    J, I = np.meshgrid(j, i)
    cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
    """

    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
    return cos_window

def gaussian2d_labels(sz,sigma):
    w,h=sz
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    center_x, center_y = w / 2, h / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    return labels

"""
max val at the top left loc
"""
def gaussian2d_rolled_labels(sz,sigma):
    w,h=sz
    xs, ys = np.meshgrid(np.arange(w)-w//2, np.arange(h)-h//2)
    dist = (xs**2+ys**2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    labels = np.roll(labels, -int(np.floor(sz[0] / 2)), axis=1)
    labels=np.roll(labels,-int(np.floor(sz[1]/2)),axis=0)
    return labels


def APCE(response_map):
    Fmax=np.max(response_map)
    Fmin=np.min(response_map)
    apce=(Fmax-Fmin)**2/(np.mean((response_map-Fmin)**2))
    return apce

def PSR(response):
    response_map=response.copy()
    max_loc=np.unravel_index(np.argmax(response_map, axis=None),response_map.shape)
    y,x=max_loc
    F_max = np.max(response_map)
    response_map[y-5:y+6,x-5:x+6]=0.
    mean=np.mean(response_map[response_map>0])
    std=np.std(response_map[response_map>0])
    psr=(F_max-mean)/std
    return psr
