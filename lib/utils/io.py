import os
import numpy as np
import json
from .bbox_helper import get_axis_aligned_bbox,cxy_wh_2_rect
import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/../.."

def get_ground_truthes_viot(img_path):
    gt_path = os.path.join(img_path, 'groundtruth.txt')
    gts=[]
    with open(gt_path, 'r') as f:
        while True:
            line = f.readline()
            if line=='':
                gts=np.array(gts,dtype=np.float32)
                #for i in range(4):  # x, y, width, height
                #    xp = range(0, gts.shape[0], 5)
                #    fp = gts[xp, i]
                #    x = range(gts.shape[0])
                #    gts[:, i] = pylab.interp(x, xp, fp)
                return gts
            if ',' in line:
                gt_pos = line.split(',')
            else:
                gt_pos=line.split()
            gt_pos_int=[(float(element)) for element in gt_pos]
            gt_pos_int = get_axis_aligned_bbox(np.array(gt_pos_int))
            target_pos = np.array([gt_pos_int[0], gt_pos_int[1]])
            target_sz = np.array([gt_pos_int[2], gt_pos_int[3]])
            location=cxy_wh_2_rect(target_pos,target_sz)
            gts.append(location)

def write_results(address, results):
    json_content = json.dumps(results, default=str)
    f = open(address + ".json", 'w')
    f.write(json_content)
    f.close()

def get_states_data(states_dir):
    st_path = os.path.join(states_dir, 'camera_states.txt')
    st = np.loadtxt(st_path, delimiter=',').astype(np.float64)
    return st[:,:]

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list

def get_ground_truthes(img_path):
    gt_path = os.path.join(img_path, 'groundtruth_rect.txt')
    gts=[]
    with open(gt_path, 'r') as f:
        while True:
            line = f.readline()
            if line=='':
                gts=np.array(gts,dtype=np.float32)

                #for i in range(4):  # x, y, width, height
                #    xp = range(0, gts.shape[0], 5)
                #    fp = gts[xp, i]
                #    x = range(gts.shape[0])
                #    gts[:, i] = pylab.interp(x, xp, fp)
                return gts
            if ',' in line:
                gt_pos = line.split(',')
            else:
                gt_pos=line.split()
            gt_pos_int=[(float(element)) for element in gt_pos]
            gts.append(gt_pos_int)

def get_ground_truthes_viot(img_path):
    gt_path = os.path.join(img_path, 'groundtruth.txt')
    gts=[]
    with open(gt_path, 'r') as f:
        while True:
            line = f.readline()
            if line=='':
                gts=np.array(gts,dtype=np.float32)

                #for i in range(4):  # x, y, width, height
                #    xp = range(0, gts.shape[0], 5)
                #    fp = gts[xp, i]
                #    x = range(gts.shape[0])
                #    gts[:, i] = pylab.interp(x, xp, fp)
                return gts
            if ',' in line:
                gt_pos = line.split(',')
            else:
                gt_pos=line.split()

            gt_pos_int=[(float(element)) for element in gt_pos]
            gt_pos_int = get_axis_aligned_bbox(np.array(gt_pos_int))
            target_pos = np.array([gt_pos_int[0], gt_pos_int[1]])
            target_sz = np.array([gt_pos_int[2], gt_pos_int[3]])
            location=cxy_wh_2_rect(target_pos,target_sz)
            gts.append(location)

def get_init_gt(img_path):
    gt_path = os.path.join(img_path, 'groundtruth_rect.txt')
    with open(gt_path, 'r') as f:
        line = f.readline()
        if ',' in line:
            gt_pos = line.split(',')
        else:
            gt_pos=line.split()
        gt_pos_int=[int(float(element)) for element in gt_pos]
    return tuple(gt_pos_int)

