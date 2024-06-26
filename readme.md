# pyTrackers
This repository contains implementation of some major correlation filter based trackers cloned from [pyCFTrackers](https://github.com/fengyang95/pyCFTrackers) repo and some deep learning trackers taken from [pytracking](https://github.com/fengyang95/pyCFTrackers). Modifications of their algorithms are made so that they can be benefit from camera state measurements. The algorithms are evaluated on a custom dataset named [VIOT-2 dataset](https://drive.google.com/file/d/1cXYGRWDcvpxmc54gyAqvfLYPxaSshgV9/view?usp=sharing). 



## Dataset
- [Link to VIOT-2 dataset](https://drive.google.com/file/d/1cXYGRWDcvpxmc54gyAqvfLYPxaSshgV9/view?usp=sharing)
- This dataset was generated with [ground_truth_generation](https://github.com/hamidrezafahimi/ground_truth_generation) repository
 

### Sample images from dataset:
- Sample images from our VIOT-2 dataset
<div>
<img src="img/dataset11.jpg"  width="130" height="100">
<img src="img/dataset12.jpg"  width="130" height="100">
<img src="img/dataset13.jpg"  width="130" height="100">
<img src="img/dataset14.jpg"  width="130" height="100">
<img src="img/dataset15.jpg"  width="130" height="100">
<img src="img/dataset16.jpg"  width="130" height="100">
<br/>
<img src="img/dataset21.jpg"  width="130" height="100">
<img src="img/dataset22.jpg"  width="130" height="100">
<img src="img/dataset23.jpg"  width="130" height="100">
<img src="img/dataset24.jpg"  width="130" height="100">
<img src="img/dataset25.jpg"  width="130" height="100">
<img src="img/dataset26.jpg"  width="130" height="100">
<br/>
<img src="img/dataset31.jpg"  width="130" height="100">
<img src="img/dataset32.jpg"  width="130" height="100">
<img src="img/dataset33.jpg"  width="130" height="100">
<img src="img/dataset34.jpg"  width="130" height="100">
<img src="img/dataset35.jpg"  width="130" height="100">
<img src="img/dataset36.jpg"  width="130" height="100">
<br/>
<img src="img/dataset41.jpg"  width="130" height="100">
<img src="img/dataset42.jpg"  width="130" height="100">
<img src="img/dataset43.jpg"  width="130" height="100">
<img src="img/dataset44.jpg"  width="130" height="100">
<img src="img/dataset45.jpg"  width="130" height="100">
<img src="img/dataset46.jpg"  width="130" height="100">
</div>


### Sample odometry:
- Gradient-descent-optimization-based Correction of
robot’s visual odometry drift utilizing ArUco markers placed
in the field.
<img src="img/sample_odometry.png" height="300">

### sample result
- Tracker precision comparison over Mixformer and
KYS trackers for pure trackers, trackers enhanced with VIOT,
and trackers enhanced with our extension
<div>
<img src="img/DiMP.png"  width="390" height="340">
<img src="img/ToMP.png"  width="390" height="340">
<img src="img/KYS.png"  width="390" height="340">
<img src="img/MixFormer.png"  width="390" height="340">
<br/>
<img src="img/Legend.png"  width="390">
</div>


## Setup

### Running in Colab Notebook

A best experience is to run the code on a cloab notebook. For running the codes please use the Google Colab notebook found [here](https://colab.research.google.com/drive/1ZpzUCRHWSrblqPyaLZDdTCLPyjdEWUBm?usp=sharing).

### Running in Local System

To run the code in a local host, first clone the repository:

#### Prerequisites

Tested on the following platform:

```
ubuntu 20.04
pytorch 1.13.1
cuda 11.7
```

#### Create virtual environment:

Having *Anaconda* installed, try creating a new conda environment:

```
conda create --name pt python=3.7.2
```

Activate the environment before the rest of package installations and also run the codes with the environment activated.

#### Install prerequsites:

Do in order:

```
pip install torch==1.13.1 torchvision
pip install matplotlib
pip install utm
pip install visdom
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.7/bin:$PATH
export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
pip install spatial-correlation-sampler
sudo apt-get install ninja-build
pip install jpeg4py
pip install timm
pip install einops
pip install lmdb
pip install opencv-python
pip install --upgrade scikit-image
pip install easydict
pip install tensorboardX
``` 

#### Download and Import Requred Data:

1. Download and unzip content of [this link](https://drive.google.com/uc?id=1xYHGJR-UWccPzbIuzMU_yf0bSkQ42qQh) and [this link](https://drive.google.com/file/d/1cXYGRWDcvpxmc54gyAqvfLYPxaSshgV9/view?usp=sharing) into `dataset/`

2. Download and import the content of [this link](https://drive.google.com/uc?id=1vuwTOMjXC3BznGUiOgT9k58ermUf3XEH) and [this link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) into `trackers/MixFormer/models/`

3. Download and import the content of [this link](https://drive.google.com/uc\?id\=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN) and [this link](https://drive.google.com/uc\?id\=1zbQUVXKsGvBEOc-I1NuGU6yTMPth_aI5) and [this link](https://drive.google.com/uc\?id\=1nJTBxpuBhN0WGSvG7Zm3yBc9JAC6LnEn) and [this link](https://drive.google.com/uc?id=1XQAtrM9n_PHQn-B2i8y6Q-PQFcAoKObA) into `trackers/pytracking/pytracking/networks/`


## Run

To run a single config:

```
#root
python launch/track/run_tracker_single.py
```

To run a set of configs on a set of sequences:

```
#root
python launch/track/run_tracker_multi.py
```
