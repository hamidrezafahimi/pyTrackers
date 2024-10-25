import cv2
from depth import seeDepth
# set torch options


import sys
from pathlib import Path

pth = str(Path(__file__).parent.resolve())
sys.path.insert(0, pth)

path_img = pth + "/input/00000001.jpg"
print (path_img)
image_frame = cv2.imread(path_img)
# cv2.imshow("asdasd", image_frame)
# cv2.waitKey(0)
output_dir = "output"
model_path = pth + "/weights/midas_v21_384.pt"
model_type = "midas_v21_384"
print (model_path)

# compute depth maps
seeDepth(image_frame, model_path, model_type)
