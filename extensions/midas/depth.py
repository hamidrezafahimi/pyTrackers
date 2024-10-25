"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from imutils.video import VideoStream
from .lib.model_loader import default_models, load_model
from pathlib import Path

first_execution = True

def read_image(img):

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

def showDepth(depth, bits=1):
    grayscale = True
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)
    
    frm = cv2.resize(out.astype("uint8"), (440, 330), interpolation = cv2.INTER_NEAREST)
    cv2.imshow("Depth", frm)
    return


def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    if optimize and device == torch.device("cuda"):
        if first_execution:
            print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                    "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                    "  half-floats.")
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    if first_execution or not use_camera:
        height, width = sample.shape[2:]
        print(f"    Input resized to {width}x{height} before entering the encoder")
        first_execution = False

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    return prediction

def seeDepth(image_frame, model_path, model_type):
    pth = str(Path(__file__).parent.resolve())
    sys.path.insert(0, pth)
    model_path = pth + "/weights/" + model_path
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


    # print("Initialize")
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device: %s" % device)
    # print (model_path)
    model, transform, net_w, net_h = load_model(device, model_path, model_type, False, None, False)

    # print("Start processing")


    # image = cv2.imread(image_name)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    # input
    original_image_rgb = read_image(image_frame)
    image = transform({"image": original_image_rgb})["image"]
    # compute
    with torch.no_grad():
        prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                False, False)

    # output
    grayscale = True
    min = prediction[0][0]
    max = prediction[0][0]
    for i in prediction:
        for j in i:
            if j < min:
                min = j
            if j > max:
                max = j
            # print(j)
    # print("min : ", min)
    # print("max : ", max)


    # Set the normalization range (e.g., a=0, b=1 for [0,1] or a=-1, b=1 for [-1,1])
    a = 0  # Minimum of desired range
    b = 5   # Maximum of desired range

    # Step 1: Find the min and max values of the array
    min_val = np.min(prediction)
    max_val = np.max(prediction)
    # print("min : ", min_val)
    # print("max : ", max_val)
    # Step 2: Normalize the array to range [a, b]
    normalized_array = (b - a) * (prediction - min_val) / (max_val - min_val) + a

    # plt.imshow(normalized_array, cmap='gray', interpolation='nearest')
    # plt.show()

    # reshaping
    prediction_temp = normalized_array
    # Step 1: Reshape the array to group into 10x10 blocks
    reshaped_array = prediction_temp.reshape(33, 10, 44, 10)

    # Step 2: Calculate the average of each 10x10 block
    average_distance = reshaped_array.mean(axis=(1, 3))

    # print(average_distance)
    
    # plt.imshow(average_distance, cmap='gray', interpolation='nearest')
    # plt.show()

    # reshaping
    prediction_temp = prediction
    # Step 1: Reshape the array to group into 10x10 blocks
    reshaped_array = prediction_temp.reshape(33, 10, 44, 10)

    # Step 2: Calculate the average of each 10x10 block
    average_distance = reshaped_array.mean(axis=(1, 3))

    # print(average_distance)
    
    # plt.imshow(average_distance, cmap='gray', interpolation='nearest')
    # plt.show()

    # cv2.imshow("Depth", prediction)
    # plt.imshow(prediction, cmap='gray', interpolation='nearest')
    # plt.show()
    showDepth(average_distance)

        

    # print("Finished")