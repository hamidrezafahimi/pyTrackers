import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(0, pth)
from lib.tracking.types import ExtType, Trackers, Datasets
from lib.tracking.run import run_once

dataset_type = Datasets.VIOT
ext_type = ExtType.kpt
# ext_type = ExtType.viot
# ext_type = ExtType.raw
data_name = "park_mavic_1"
start_frame = 1
end_frame = 1650
# tracker_type = Trackers.MIXFORMER
tracker_type = Trackers.DIMP
# tracker_type = Trackers.TOMP
# tracker_type = Trackers.KYS
# tracker_variant = 'tomp101'
# tracker_variant = 'vit'
# tracker_variant = 'prdimp50'
tracker_variant = 'dimp50'
# tracker_variant = 'default'
data_path = pth + "/dataset/VIOT/" + data_name

if __name__ == '__main__':
    run_once(data_path=data_path, tracker_type=tracker_type, tracker_variant=tracker_variant,
             ext_type=ext_type, start_frame=start_frame, end_frame=end_frame, 
             dataset_type=dataset_type)
