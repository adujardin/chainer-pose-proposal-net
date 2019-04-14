# Real Time 3D Pose detection

Pose proposal network provides the 2D pose using a single image, similarly to [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), but much faster reaching real time speed on embedded CPU (see original work, running on a [Raspberry Pi Zero](https://twitter.com/9_ties/status/1059750417679114240)).

This project integrates the ZED SDK into the 2D pose estimation to provides a real time 3D pose estimation, much faster than [OpenPose](https://github.com/adujardin/zed-openpose)

## Using the ZED stereo camera

While the network can be run on CPU or GPU and doesn't require CUDA, the ZED SDK does. Therefore, this sample requires an Nvidia graphics card with CUDA, the detection is also obviously much faster on GPU.

1. Install the required dependencies:
    
    1. [ZED SDK](https://www.stereolabs.com/developers/release/) and the [Python API](https://github.com/stereolabs/zed-python-api)
    2. Chainer + Cupy


2. Download a pretrained model from Idein [here](https://github.com/Idein/chainer-pose-proposal-net/releases), or retrain one (see below).

3. Run `high_speed_zed.py`, for instance using resnet18 :

```bash
    python3 high_speed_zed.py result/resnet18_384x384_coco/
```

## DISCLAIMER

This is **work in progress**, the 3D estimation isn't robust as it directly reads the raw value in the depth map and doesn't deal with invalid values yet. The goal was to start with the ZED SDK integration, then 3D display with limbs and kp in color (this is the current state), then a more robust 3D estimation.


## Original Repo : 

- https://github.com/Idein/chainer-pose-proposal-net

- This is an (unofficial) implementation of [Pose Proposal Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sekii_Pose_Proposal_Networks_ECCV_2018_paper.pdf) with Chainer including training and prediction tools.

Please cite the paper in your publications if it helps your research:

    @InProceedings{Sekii_2018_ECCV,
      author = {Sekii, Taiki},
      title = {Pose Proposal Networks},
      booktitle = {The European Conference on Computer Vision (ECCV)},
      month = {September},
      year = {2018}
      }

### License

Copyright (c) 2018 Idein Inc. & Aisin Seiki Co., Ltd.
All rights reserved.

This project is licensed under the terms of the [license](LICENSE).


## Demo: Realtime Pose Estimation

We tested on an Ubuntu 16.04 machine with GPU GTX1080(Ti)

### Build Docker Image for Demo

We will build OpenCV from source to visualize the result on GUI.

```
$ cd docker/gpu
$ cat build.sh
docker build -t ppn .
$ sudo bash build.sh
```

### Run using a webcam

- Set your USB camera that can recognize from OpenCV.

- Using the feature of [Static Subgraph Optimizations](http://docs.chainer.org/en/stable/reference/static_graph_design.html) to accelerate inference speed, (requires Chainer 5.y.z and CuPy 5.y.z e.g. 5.0.0 or 5.1.0)

- Run `high_speed.py`

```
$ python high_speed.py ./trained
```

## Training

- Prior to training, let's download dataset. You can train with MPII or COCO dataset by yourself.
- For simplicity, we will use docker image of [idein/chainer](https://hub.docker.com/r/idein/chainer/) which includes Chainer, ChainerCV and other utilities with CUDA driver. This will save time setting development environment.
- For more information see:
  - https://docs.docker.com/install/
  - https://github.com/NVIDIA/nvidia-docker

### Prepare Dataset

#### MPII

- If you train with COCO dataset you can skip.
- Access [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/) and jump to `Download` page. Then download and extract both `Images (12.9 GB)` and `Annotations (12.5 MB)` at `~/work/dataset/mpii_dataset` for example.

##### Create `mpii.json`

We need decode `mpii_human_pose_v1_u12_1.mat` to generate `mpii.json`. This will be used on training or evaluating test dataset of MPII.

```
$ sudo docker run --rm -v $(pwd):/work -v path/to/dataset:mpii_dataset -w /work idein/chainer:4.5.0 python3 convert_mpii_dataset.py mpii_dataset/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat mpii_dataset/mpii.json
```

It will generate `mpii.json` at `path/to/dataset`. Where `path/to/dataset` is the root directory of MPII dataset, for example, `~/work/dataset/mpii_dataset`. For those who hesitate to use Docker, you may edit `config.ini` as necessary.

#### COCO

- If you train with MPII dataset you can skip.
- Access [COCO dataset](http://cocodataset.org/) and jump to `Dataset` -> `download`. Then download and extract `2017 Train images [118K/18GB]`, `2017 Val images [5K/1GB]` and `2017 Train/Val annotations [241MB]` at `~/work/dataset/coco_dataset:/coco_dataset` for example.

### Running Training Scripts

OK let's begin!

```
$ cat begin_train.sh
cat config.ini
docker run --rm \
-v $(pwd):/work \
-v ~/work/dataset/mpii_dataset:/mpii_dataset \
-v ~/work/dataset/coco_dataset:/coco_dataset \
--name ppn_idein \
-w /work \
idein/chainer:5.1.0 \
python3 train.py
$ sudo bash begin_train.sh
```

- Optional argument `--runtime=nvidia` maybe require for some environment.
- It will train a model the base network is MobileNetV2 with MPII dataset located in `path/to/dataset` on host machine.
- If we would like to train with COCO dataset, edit a part of `config.ini` as follow:

before

```
# parts of config.ini
[dataset]
type = mpii
```

after

```
# parts of config.ini
[dataset]
type = coco
```

- We can choice ResNet based network as the original paper adopts. Edit a part of `config.ini` as follow:

before

```
[model_param]
model_name = mv2
```

after

```
[model_param]
# you may also choice resnet34 and resnet50
model_name = resnet18
```

## Prediction

- Very easy, all we have to do is, for example:

```
$ sudo bash run_predict.sh ./trained
```

- If you would like to configure parameter or hide bounding box, edit a part of `config.ini` as follow:

```
[predict]
# If `False` is set, hide bbox of annotation other than human instance.
visbbox = True
# detection_thresh
detection_thresh = 0.15
# ignore human its num of keypoints is less than min_num_keypoints
min_num_keypoints= 1
```