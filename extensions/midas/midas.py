import torch
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from .lib.model_loader import default_models, load_model
from pathlib import Path


class Midas():
    def __init__(self, model_path, model_type):
        self.first_execution = True
        self.pth = str(Path(__file__).parent.resolve())
        sys.path.insert(0, self.pth)
        self.model_path = self.pth + "/weights/" + model_path
        self.model_type = model_type
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, self.model_path, self.model_type, False, None, False)

    def read_image(self, img):

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        return img
    
    def showDepth(self, window_name, depth, bits=1):
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
        cv2.imshow(window_name, frm)
        return
    
    def process(self, device, model, model_type, image, input_size, target_size, optimize, use_camera):
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
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if self.first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                        "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                        "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if self.first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            self.first_execution = False

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

    def getNormalized(self, min, max, data):
        # Set the normalization range (e.g., min=0, max=1 for [0,1] or min=-1, max=1 for [-1,1])
        # min      Minimum of desired range
        # max      Maximum of desired range

        # Find the min and max values of the array
        min_val = np.min(data)
        max_val = np.max(data)

        # Normalize the array to range [min, max]
        return ((max - min) * (data - min_val) / (max_val - min_val) + min)


    def seeDepth(self, image_frame):
        # input
        original_image_rgb = self.read_image(image_frame)
        image = self.transform({"image": original_image_rgb})["image"]
        # compute
        with torch.no_grad():
            prediction = self.process(self.device, self.model, self.model_type, image, (self.net_w, self.net_h), original_image_rgb.shape[1::-1],
                                    False, False)

        # Normalize the array to range [a, b]
        normalized_array = self.getNormalized(0, 255, prediction)

        # reshape_matrix_by_percentage
        reshaped_array = self.reshape_matrix_by_percentage(normalized_array, 2, 2)

        self.showDepth("Depth", reshaped_array)
        return reshaped_array
        

    def reshape_matrix_by_percentage(self, matrix, height_percentage, width_percentage):
        height, width = matrix.shape
        
        # Calculate the block size for sub-matrices based on the percentage
        sub_height = max(1, int(height * height_percentage / 100))
        sub_width = max(1, int(width * width_percentage / 100))

        # Calculate the number of sub-matrices in the new matrix
        new_height = (height + sub_height - 1) // sub_height  # ceil division to handle remainder pixels
        new_width = (width + sub_width - 1) // sub_width      # ceil division to handle remainder pixels

        # Initialize a new matrix to store the averages of the sub-matrices
        new_matrix = np.zeros((new_height, new_width))

        for i in range(new_height):
            for j in range(new_width):
                # Define the region in the original matrix
                start_i = i * sub_height
                start_j = j * sub_width
                end_i = min((i+1) * sub_height, height)  # Handle remainder pixels on the right/bottom
                end_j = min((j+1) * sub_width, width)

                # Extract the sub-matrix for the current section
                sub_matrix = matrix[start_i:end_i, start_j:end_j]
                
                # Compute the mean of the sub-matrix and assign it to the new matrix
                new_matrix[i, j] = np.mean(sub_matrix)

        return new_matrix
    
    def writeDepth(self, depth, path):
        bits = 1
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2**(8*bits))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)

        frm = cv2.resize(out.astype("uint8"), (440, 330), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(path, frm)
        return