import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/.."
sys.path.insert(2, pth)
import numpy as np
from pytracker import PyTracker
from lib.utils.io import get_img_list,get_states_data,get_ground_truthes_viot,write_results
from lib.tracking.types import ExtType, Trackers


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
    config_dict['ext_type'] = ExtType.viot

    # Extract ground-truth data
    gts = get_ground_truthes_viot(data_path)
    # Crop ground-truth data sequence into init- and end-frames
    gts = gts[config_dict['start_frame'] - 1:config_dict['end_frame']]
    # Convert gt data to int
    viot_results_gts = [list(gt.astype(np.int)) for gt in gts]

    frame_list = get_img_list(data_path)
    frame_list.sort()
    states = get_states_data(data_path) ## VIOT

    tracker_mixformer = PyTracker(data_path, _gts=gts, tracker_title=Trackers.MIXFORMERVIT,
                                  dataset_config=config_dict, fl=frame_list, sts=states, 
                                  ext_type=ExtType.viot)
    
    mixformer_preds, save_phrase = tracker_mixformer.tracking(data_name=data_name, verbose=True)
    mixformer_results = {}
    mixformer_results[data_name] = {}
    mixformer_results[data_name]['tracker_mixformer_preds'] = []
    for mixformer_pred in mixformer_preds:
        mixformer_results[data_name]['tracker_mixformer_preds'].append(list(mixformer_pred.astype(np.int)))
    write_results(save_phrase, mixformer_results)
    print('mixformer done!')
