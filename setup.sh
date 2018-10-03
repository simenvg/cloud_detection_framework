#!/bin/bash


# Stuff
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git
sudo apt-get install gcc
sudo apt-get install make


# CUDA
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-9-0


# CUDNN
wget https://www.dropbox.com/s/67p1cm243ebxmjy/cudnn-9.0-linux-x64-v7.2.1.38.tgz?dl=0
mv cudnn-9.0-linux-x64-v7.2.1.38.tgz?dl=0 cudnn-9.0-linux-x64-v7.2.1.38.tgz
tar xzvf cudnn-9.0-linux-x64-v7.2.1.38.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*


# Darknet
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
git clone https://github.com/pjreddie/darknet
cd darknet
sed -i 's/GPU=.*/GPU=1/' Makefile
sed -i 's/CUDNN=.*/CUDNN=1/' Makefile
make
cd ..
cd mkdir data