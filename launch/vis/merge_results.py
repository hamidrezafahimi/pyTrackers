import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(0, pth)
import json
import os

viot_results = {}

path = pth + "/results/"
dir_list = os.listdir(path)

for file_name in dir_list:

    if ".txt" in file_name or ".mp4" in file_name:
        continue
    if not ".json" in file_name:
        continue
    result_json_path = path + file_name
         
    f = open(result_json_path, 'r')
    results = json.load(f)

    for key in results.keys():

        if not key in viot_results.keys():
            viot_results[key] = {}

        for k in results[key].keys():

            if not k in viot_results[key].keys():
                viot_results[key][k] = []

            viot_results[key][k] = results[key][k]

json_content = json.dumps(viot_results, default=str)
f = open(pth + '/results/all_results.json', 'w')
f.write(json_content)
f.close()