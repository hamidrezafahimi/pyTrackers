import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/.."
sys.path.insert(0, pth)
from lib.utils.io import get_gt, get_tracker_config, get_run_config, write_gt
from .pytracker import PyTracker


def run_once(data_path, tracker_type, tracker_variant, dataset_type, ext_type, start_frame, end_frame):
    data_name = Path(data_path).stem
    tt = get_tracker_config(tracker_type, tracker_variant)
    config_dict = get_run_config(data_name=data_name, ds_type=dataset_type, ext_type=ext_type,
                                 tr=[tracker_type, tracker_variant], start_frame=start_frame,
                                 end_frame=end_frame)
    frame_list, states, init_gt, gts = get_gt(data_path, config_dict['start_frame'],
                                              config_dict['end_frame'])
    write_gt(gts, states, data_path)
    tracker = PyTracker(data_path, tracker_title=tt, dataset_config=config_dict, 
                        ext_type=ext_type, verbose=True)
    tracker.tracking(data_name=data_name, frame_list=frame_list, init_gt=init_gt, 
                     states=states)