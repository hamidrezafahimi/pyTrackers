import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/.."
sys.path.insert(2, pth)
import numpy as np
from pytracker import PyTracker
from lib.utils.io import get_img_list,get_states_data,get_ground_truthes_viot,write_results
from lib.tracking.types import ExtType, Trackers

# def cropList(lst, idx_i, idx_f):
#     return lst[idx_i-1:idx_f]

if __name__ == '__main__':
    config_dict = {}
    data_name = 'pm1'
    data_path = pth + "/dataset/VIOT/pm1"
    config_dict['start_frame'] = 110
    config_dict['end_frame'] = 191
    config_dict['fov'] = 66.0
    config_dict['ratio_tresh'] = {}
    config_dict['ratio_tresh']['MIXFORMER_VIT'] = 0.995
    config_dict['interp_factor'] = {}
    config_dict['interp_factor']['MIXFORMER_VIT'] = 1.0
    ext = ExtType.viot
    config_dict['ext_type'] = ext

    # Extract ground-truth data
    gts = get_ground_truthes_viot(data_path)
    # Crop ground-truth data sequence into init- and end-frames
    gts = gts[config_dict['start_frame'] - 1:config_dict['end_frame']]
    # # Convert gt data to int
    # viot_results_gts = [list(gt.astype(np.int)) for gt in gts]
    frame_list = get_img_list(data_path)
    frame_list.sort()
    frame_list = frame_list[config_dict['start_frame']-1:config_dict['end_frame']]
    states = get_states_data(data_path) ## VIOT
    states = states[config_dict['start_frame']-1:config_dict['end_frame']]

    tt = Trackers.MIXFORMER
    tcfg = {}
    tcfg['type'] = 'vit'
    tcfg['model'] = 'mixformer_vit_base_online.pth.tar'
    tcfg['search_area_scale'] = 4.0
    tcfg['yaml_name'] = 'baseline'
    tcfg['dataset_name'] = 'got10k_test'
    tt.config = tcfg
    tracker_mixformer = PyTracker(data_path, tracker_title=tt, dataset_config=config_dict,  
                                  ext_type=ExtType.viot, verbose=True)

    save_phrase = pth + "/results/{:s}_{:s}_".format(tt.name, data_name) + ext.name

    tracker_mixformer.tracking(data_name=data_name, frame_list=frame_list, 
                                                 init_gt=gts[0], states=states)
