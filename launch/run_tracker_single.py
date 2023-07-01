import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/.."
sys.path.insert(0, pth)
from lib.utils.io import get_gt, get_tracker_config, get_run_config
from lib.tracking.types import ExtType, Trackers, Datasets
from lib.tracking.pytracker import PyTracker

dataset_type = Datasets.VIOT
ext_type = ExtType.raw
data_name = "park_mavic_1"
start_frame = 110
end_frame = 191
tracker_type = Trackers.MIXFORMER
tracker_variant = 'vit'
data_path = pth + "/dataset/VIOT/" + data_name

if __name__ == '__main__':
    tt = get_tracker_config(tracker_type, tracker_variant)
    config_dict = get_run_config(data_name=data_name, ds_type=dataset_type, ext_type=ext_type,
                                 tr=[tracker_type, tracker_variant], start_frame=start_frame,
                                 end_frame=end_frame)
    frame_list, states, init_gt = get_gt(data_path, config_dict)
    tracker = PyTracker(data_path, tracker_title=tt, dataset_config=config_dict, 
                        ext_type=ext_type, verbose=True)
    tracker.tracking(data_name=data_name, frame_list=frame_list, init_gt=init_gt, 
                     states=states)
