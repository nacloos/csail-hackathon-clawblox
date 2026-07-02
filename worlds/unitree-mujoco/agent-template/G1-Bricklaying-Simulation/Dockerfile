# CUDA 12.8 base image
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    git cmake curl wget bzip2 ca-certificates vim \
    python3.10 python3-pip iproute2 iputils-ping \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH=

# Configure conda and install pinocchio
RUN conda config --set channel_priority flexible && \
    echo "yes" | conda tos accept && \
    conda install -y python=3.10 && \
    echo "yes" | conda tos accept && \
    conda install -y -c conda-forge pinocchio && \
    conda clean --all --yes

# Install PyTorch + other pip dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install opencv-python open3d transformers
RUN pip install ros-rclpy

# Set working directory
WORKDIR /ws

# CycloneDDS install
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds.git && \
    cd cyclonedds && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=ON .. && \
    cmake --build . --target install

# Unitree SDK
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && \
    pip install -e unitree_sdk2_python

# Grounded SAM-2
RUN git clone https://github.com/IDEA-Research/Grounded-SAM-2.git && \
    pip install -e Grounded-SAM-2 && \
    pip install --no-build-isolation -e Grounded-SAM-2/grounding_dino && \
    cd Grounded-SAM-2/checkpoints && bash download_ckpts.sh && \
    cd ../gdino_checkpoints && bash download_ckpts.sh
