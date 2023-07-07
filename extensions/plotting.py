from matplotlib import pyplot as plt
import numpy as np
import json




# cam_pos = gps_to_ned(_ref_loc, states[1:4])
# imu_meas = states[4:7]
# body_dir = cam_to_body(rect)
# inertia_dir = body_to_inertia(body_dir, imu_meas)
# target_pos = scale_vector(inertia_dir, cam_pos[2]) + cam_pos

result_json_path = '/home/hamid/w/pyTrackers/results/gts_park_mavic_1.json'
f = open(result_json_path, 'r')
gt = json.load(f)
gtnp = np.array(gt["park_mavic_1"]["gts"])
print(gtnp.shape)

df = np.loadtxt('/home/hamid/w/pyTrackers/results/MIXFORMER_park_mavic_1_viot_pose3ds.txt', delimiter=',')
print(df.shape)