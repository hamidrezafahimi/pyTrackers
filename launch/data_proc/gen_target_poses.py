import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(0, pth)
from lib.utils.io import get_gt, write_target_poses_ned

data_names = {
    "park_mavic_1":[1,1005],
    "park_mavic_2":[250,945],
    "park_mavic_3":[1,1800],
    "park_mavic_4":[1,1900],
    "park_mavic_5":[1,1697],
    "park_mavic_6":[1,1137]
    # "park_mavic_7":[1,360]
    # "soccerfield_mavic_3":[1,500],
    # "soccerfield_mavic_4":[500,1297]
    }

if __name__ == '__main__':
    for d_name in data_names:
        data_path = pth + "/dataset/VIOT/" + d_name
        frame_list, states, init_gt, gts = get_gt(data_path, data_names[d_name][0],
                                                  data_names[d_name][1])
        write_target_poses_ned(data_path, d_name, states)
