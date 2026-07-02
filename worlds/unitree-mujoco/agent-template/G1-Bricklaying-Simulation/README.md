# Bricklaying with Unitree G1

Autonomony stack for bricklaying with the Unitree G1.  Contains:
- modules for robotic perception, planning, and control 
- nodes for real-time communication with G1 
- top-level scripts for functionality demonstration
- Dockerfile and docker-compose for containerized use

Most of this code is intended to be run on an external machine, e.g. the living lab "ground station" computer.  Real-time communication with the G1 is handled via CycloneDDS, assuming the robot and the user computer have been networked correctly.  The current configuration is to connect everything through the G1-Router via ethernet.

# Media

TODO gif

# Installation

Clone this repository
```
mkdir ~/drl
cd drl
git clone https://github.com/Distributed-Robotics-Lab/g1-bricklaying.git
```

Build the Docker image (if it doesn't already exist)
```
cd g1-bricklaying
docker build -t g1-bricklaying .
```

# Usage

1. The perception module requires real-time images from the G1 realsense camera.  So, start a dedicated publisher node on the G1 to send these images over the network.

Ssh into the G1
```
ssh unitree@192.168.123.164  # password: 123
```

Source the G1's configured python environment
```
conda activate unitree
```

Start up the image publisher, specifying the network interface as `eth0` or whatever is the correct ip connection
```
cd drl/image_publisher
python image_publisher.py eth0
```

2. Now we can start running code on the user's machine (external computer), which will listen to these images.

Spin up the docker container
```
docker compose up -d
docker compose exec g1-bricklaying /bin/bash
```

Run the main bricklaying script, specifying the network interface as `eno1` or whatever is the correct ip connection
```
python g1-bricklaying/scripts/pick_and_place.py eno1
```

# Demo instructions

The following is a set of instructions for running the pick and place demo.  We start by assuming the G1 is turned off, in the harness, everything is networked/connected properly through the router, and the living lab ground station has already been installed and configured to use this codebase.

1. Move G1's arms to a safe spot for DEX3 hand initialization.

![hand_open](./src/g1_bricklaying/assets/readme/hand_open.png)

2. Turn on the G1 and allow DEX3 hands to initialize.  When it's finished turning on, you should hear the G1 audibly give the voice command for "zero torque mode" or "damping mode".

3. Manually adjust the DEX3 hands to a neatly balled up fist.

![fist_front](./src/g1_bricklaying/assets/readme/fist_front.png)

4. Manually adjust the G1 arms to be at the sides of the hip.

![fist_side](./src/g1_bricklaying/assets/readme/fist_side.png)

5. (The above instructions were to ensure the fingers don't get jammed when the G1 moves into "ready mode" or "motion mode").  Now that these precautions have been taken, use the remote control to enter "damping mode" (L2 + B) and then subsequently "ready mode" (L2 + Up).

6. Lower the harness so the G1's feet are touching the floor and its weight is not being placed on the harness.  Then, use the remote control to enter "motion mode" (R1 + X).

7. Take the dog off its leash.  The G1 can now be walked around with the joystick.

![leash](./src/g1_bricklaying/assets/readme/leash.png)

8. Walk the G1 over to the pick and place demo table.  (Assuming you're familiar with the table setup and brick organization that the demo is expecting).  The demo is sensitive to where the G1 is with respect to the table: it should be close enough that it can easily reach the placing zone, but not so close that it would cram/jam its arms within the picking zone.

![walk](./src/g1_bricklaying/assets/readme/walk.png)

9. Following the instructions in the ``Usage`` section above, run the demo.

10. Walk the G1 back to the harness, hoist up the harness so there is no slack, then enter "damping mode" (L2 + B).

# Notes

## Coordinate systems

Coordinate notation:
```
R_A_to_B    # rotation from frame A to frame B
p_A_to_B_W  # position from A to B with respect to frame W
```
Note that left matrix multiplication with the rotation matrix, R_A_to_B @ x, rotates a vector x from frame B to frame A

## Installing Pytorch on Jetson

Torch 2.0.0 for Jetpack 5.1 on python3.8
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
```
conda create -n <name> python=3.8
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
python3 -m pip install --no-cache $TORCH_INSTALL
```

Torchvision 0.15.1 from source (compatible with above torch version)
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```
git clone --branch release/0.15 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.15.1  
python3 setup.py install --user
```

May need to also install pybind11:
```
pip install pybind11
```

Other installs:
```
pip install ultralytics
pip install open3d
conda install -c conda-forge pinocchio  # important to install via conda-forge, not pip
```

## Debug notes for conda env setup

Ultralytics can't import FastSAM:
```
pip install --force-reinstall --no-cache-dir charset-normalizer
```

ImportError: /lib/aarch64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/unitree/miniforge3/envs/pick_26-02-28/lib/python3.8/site-packages/scipy/spatial/_ckdtree.cpython-38-aarch64-linux-gnu.so)
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```