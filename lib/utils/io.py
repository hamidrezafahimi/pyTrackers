import os
import numpy as np
import json
from .bbox_helper import get_axis_aligned_bbox,cxy_wh_2_rect
from pathlib import Path
from lib.tracking.types import Datasets
import yaml
from .gt import FlatGroundTargetLocator
from .geom import all_wrt_ref_ned
root_path = str(Path(__file__).parent.resolve()) + "/../.."
config_path = root_path + "/config"

def write_target_poses_ned(d_path, d_name, sts):
    target_gt_path = d_path + "/target_poses.txt"
    fgtl = FlatGroundTargetLocator(target_gt_path, sts[0,1:4])
    target_poses_ned = fgtl.calc_target_ned_poses(sts)
    tpf = open(root_path + "/results/" + d_name + "_target_poses.txt", 'w')
    np.savetxt(tpf, target_poses_ned, delimiter=", ")
    tpf.close()

def write_gt(gts, sts, d_path):#, write_target_gt=True):
    d_name = Path(d_path).stem
    write_target_poses_ned(d_path, d_name, sts)
    csf = open(root_path + "/results/"+d_name+"_cam_states.txt", 'w')
    camera_poses_ned = all_wrt_ref_ned(sts)
    np.savetxt(csf, camera_poses_ned, delimiter=", ")
    csf.close()
    gt_dict = {}
    gt_dict[d_name] = {}
    pses = []
    for gt in gts:
        pses.append(list(gt.astype(np.int)))
    gt_dict[d_name]["gts"] = pses
    f = open(root_path + "/results/gts_"+d_name+".json", 'w')
    json_content = json.dumps(gt_dict, default=str)
    f.write(json_content)
    f.close()

def get_run_config(data_name, ds_type, ext_type, tr, start_frame, end_frame):
    config_dict = {}
    # data_name = Path(path).stem
    if ds_type == Datasets.CUSTOM:
        pass
        # with open(path + "/config.yaml", 'r') as f:
        #     config = yaml.full_load(f)
        #     ...
    elif ds_type == Datasets.VIOT:
        with open(config_path + "/viot/config.yaml", 'r') as f:
            config = yaml.full_load(f)
            config_dict['fov'] = config['fov'][data_name]
            config_dict['ratio_thresh'] = config['ratio_thresh'][tr[0].name + '_' + tr[1]]
            config_dict['interp_factor'] = \
                config['interp_factor'][tr[0].name + '_' + tr[1]][data_name]
    else:
        raise Exception("Unsupported dataset type")
    config_dict['start_frame'] = start_frame
    config_dict['end_frame'] = end_frame
    config_dict['ext_type'] = ext_type
    return config_dict

def get_gt(data_path, start_frame, end_frame):
    gts = get_ground_truthes_viot(data_path)
    frame_list = get_img_list(data_path)
    frame_list.sort()
    states = get_states_data(data_path)
    return frame_list[start_frame-1:end_frame], states[start_frame-1:end_frame], \
           gts[start_frame - 1] , gts[start_frame-1:end_frame]

def get_tracker_config(ttype, variant):
    tobj = ttype
    with open(root_path + "/config/trackers/"+ tobj.name +".yaml", 'r') as f:
        config = yaml.full_load(f)
        tobj.config = config[variant]
    return tobj

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

