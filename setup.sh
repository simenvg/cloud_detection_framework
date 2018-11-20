#!/bin/bash

cd ~

# Stuff
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install git
sudo apt-get -y install gcc
sudo apt-get -y install make
sudo apt-get -y install unzip
sudo apt-get -y install python-pip


export LC_ALL=en_US.UTF-8
echo  'export LC_ALL=en_US.UTF-8' >> ~/.bashrc 


# CUDA
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda-9-0



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
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc 
echo  'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc 
git clone https://github.com/pjreddie/darknet
cd darknet
wget https://pjreddie.com/media/files/darknet53.conv.74
sed -i 's/GPU=.*/GPU=1/' Makefile
sed -i 's/CUDNN=.*/CUDNN=1/' Makefile
sed -i -e 's/        if(i%10000==0 || (i < 1000 && i%100 == 0)){/        if(i%400==0){/g' ./examples/detector.c
sed -i -e "s|#lib = CDLL(\"/home/pjreddie/documents/darknet/libdarknet.so\", RTLD_GLOBAL)|lib = CDLL(\"$HOME/darknet/libdarknet.so\", RTLD_GLOBAL)|g" ./python/darknet.py
sed -i -e "s|lib = CDLL(\"libdarknet.so\", RTLD_GLOBAL)|#lib = CDLL(\"libdarknet.so\", RTLD_GLOBAL)|g" ./python/darknet.py
make

# Protoc
cd ~
curl -OL https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-linux-x86_64.zip
unzip protoc-3.4.0-linux-x86_64.zip -d protoc3
sudo mv protoc3/bin/* /usr/local/bin/
sudo mv protoc3/include/* /usr/local/include/

# Tensorflow
cd ~
sudo apt -y install python3-dev python3-pip
sudo pip3 install -U virtualenv  # system-wide install
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate  # sh, bash, ksh, or zsh
pip install --upgrade pip
pip install --upgrade tensorflow-gpu
pip install pandas
pip install lxml
pip install pillow
pip install opencv-python

# Tensorflow object detection API
cd ~
git clone https://github.com/tensorflow/models.git 
cd models/research
python setup.py build
python setup.py install
sed -i -e 's/          eval_config, category_index.values(), eval_dict)/          eval_config, list(category_index.values()), eval_dict)/g' ./object_detection/model_lib.py
sed -i '27itf.logging.set_verbosity(tf.logging.INFO)' ./object_detection/model_main.py
pip install pycocotools
echo 'export PYTHONPATH=$PYTHONPATH:~/models/research:~/models/research/slim' >> ~/.bashrc 
echo  'protoc -I=$HOME/models/research $HOME/models/research/object_detection/protos/*.proto --python_out=$HOME/models/research' >> ~/.bashrc 

deactivate

pip install opencv-python

# Setup data folder
cd ~
mkdir data
cd data
mkdir datasets
mkdir SSD_mobilenet
cd SSD_mobilenet
mkdir data
mkdir models
cd models
mkdir model
cd model
wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar xzvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz














