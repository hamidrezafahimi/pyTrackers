import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/.."
sys.path.insert(0, pth)
from lib.tracking.types import ExtType, Trackers, Datasets
from lib.tracking.run import run_once

dataset_type = Datasets.VIOT
ext_type = ExtType.viot

data_names = {
    # "cup_0.5HZ":[1,220],
    # "cup_0.9HZ":[1,760],
    # "cup_1.1HZ":[1,329],
    # "cup_1.5HZ":[1,312],
    # "cup_1.8HZ":[1,357],
    # "cup_2.1HZ":[1,465],
    # "cup_3.2HZ":[1,254],
    "park_mavic_1":[110,191]
    # "park_mavic_1":[1,1005],
    # "park_mavic_2":[45,945],
    # "park_mavic_3":[710,1100],
    # "park_mavic_4":[1,500],
    # "park_mavic_5":[840,1697],
    # "park_mavic_6":[1,1137],
    # "park_mavic_7":[1,360],
    # "soccerfield_mavic_3":[1,500],
    # "soccerfield_mavic_4":[500,1297]
    }

tr_lists = [
    # [Trackers.KCF, 'hog'],
    # [Trackers.LDES, 'default'],
    # [Trackers.STRCF, 'default'],
    # [Trackers.CSRDCF, 'default'],
    # [Trackers.DIMP, 'dimp50'],
    # [Trackers.DIMP, 'prdimp50'],
    # [Trackers.KYS, 'default'],
    # [Trackers.TOMP, 'default'],
    [Trackers.MIXFORMER, 'vit']
]

if __name__ == '__main__':

    for tr_list in tr_lists:
        for d_name in data_names:
            data_path = pth + "/dataset/VIOT/" + d_name
            run_once(data_path=data_path, tracker_type=tr_list[0], 
                     tracker_variant=tr_list[1], ext_type=ext_type, 
                     start_frame=data_names[d_name][0], end_frame=data_names[d_name][0])