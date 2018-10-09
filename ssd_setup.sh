# CAFFE INSTALLATION: http://caffe.berkeleyvision.org/installation.html

# Keep Ubuntu or Debian up to date
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y autoremove

# DEPENDENCIES
sudo apt-get install -y libopenblas-dev
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y libprotobuf-dev protobuf-compiler
sudo apt-get install -y libgoogle-glog-dev
sudo apt-get install -y libgflags-dev
sudo apt-get install -y libhdf5-dev
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y liblmdb-dev
sudo apt-get install -y libleveldb-dev
sudo apt-get install -y libsnappy-dev
sudo apt-get install -y libprotoc-dev


# INTERFACES (Python 3)
sudo apt-get install -y python3-dev python3-numpy libboost-python-dev

# CLONING AND COMPILING
git clone https://github.com/weiliu89/caffe.git
cd caffe
git checkout ssd
# git clone https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
sed -i -e 's/# USE_CUDNN := 1/USE_CUDNN := 1/g' Makefile.config
sed -i -e 's/# USE_OPENCV := 0/USE_OPENCV := 0/g' Makefile.config
sed -i -e 's/CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \\/CUDA_ARCH := #-gencode arch=compute_20,code=sm_20 \\/g' Makefile.config
sed -i -e 's/		-gencode arch=compute_20,code=sm_21 \\/		#-gencode arch=compute_20,code=sm_21 \\/g' Makefile.config
sed -i -e 's/		-gencode arch=compute_20,code=sm_21 \\/		#-gencode arch=compute_20,code=sm_21 \\/g' Makefile.config
sed -i -e 's/$(ANACONDA_HOME)\/include\/python2.7 \\/#$(ANACONDA_HOME)\/include\/python2.7 \\/g' Makefile.config
sed -i -e 's/$(ANACONDA_HOME)\/lib\/python2.7\/site-packages\/numpy\/core\/include \\/#$(ANACONDA_HOME)\/lib\/python2.7\/site-packages\/numpy\/core\/include \\/g' Makefile.config
sed -i -e 's/INCLUDE_DIRS := $(PYTHON_INCLUDE) \/usr\/local\/include/INCLUDE_DIRS := $(PYTHON_INCLUDE) \/usr\/local\/include \/usr\/include\/hdf5\/serial/g' Makefile.config
sed -i -e 's/LIBRARY_DIRS := $(PYTHON_LIB) \/usr\/local\/lib \/usr\/lib/LIBRARY_DIRS := $(PYTHON_LIB) \/usr\/local\/lib \/usr\/lib \/usr\/lib\/x86_64-linux-gnu \/usr\/lib\/x86_64-linux-gnu\/hdf5\/serial/g' Makefile.config





# Adjust Makefile.config (for example, if using Anaconda Python)
# make all -j8
# make test -j8
# make runtest
#
# make all matcaffe
# make mattest
# 
# PERSONAL NOTES AND PERSONAL CONFIGURATION
# CPU_ONLY := 1
# OPENCV_VERSION := 3
# BLAS := open
# BLAS_LIB := /usr/lib/openblas-base
# MATLAB_DIR := /opt/MATLAB/R2015a

# INSTALL OPENCV 3 with -DWITH_GDAL=OFF!!! Error OpenCV 3 with GDAL.