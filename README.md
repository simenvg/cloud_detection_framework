# Cloud Detection Framework

This project is a part of my master thesis and serves as a framework for using Googles cloud computing engine to easily train and test object detectors on custom data.

## Getting Started

### Prerequisites

What things you need to install the software and how to install them

```
 * Setup a Google cloud instance with GPU and Ubuntu 16.04. This is tested for Nvidia Tesla K80. [See Googles get started guide](https://cloud.google.com/compute/docs/instances/create-start-instance)
 * Download and install the [gcloud SDK and command line tool](https://cloud.google.com/sdk/install)
```

### Installing

When you have installed the gcloud SDK and made an instance on Google Cloud, it can be accessed with SSH

```
gcloud compute ssh <instance-name> --zone <zone>
```

Clone this git repo and run setup.sh

```
./cloud_detection_framework/setup.sh
```
This will:
 * Install CUDA and cudnn
 * Download and build Darknet (YOLO) with GPU support
 * Install Tensorflow and its object detection API
 * Add some paths and other stuff to .bashrc 
 * Install a lot of stuff because dependencies
 * Setup a directory structure
 * Download pretrained weights for SSD and YOLOv3 

After setup.sh has finished run
```
. .bashrc
```
from your home directory to get your paths correct


## Uploading Data

To begin training we need some data. The data must be labelled in the Voc Pascal format, the xml files and the jpg files should be in the same folder. To label images [LabelImg](https://github.com/tzutalin/labelImg) which will label the images in the correct format. 

During the setup a "data" directory was made in the home directory. In data's subdirectory "datasets" the data should be uploaded. It should look like this.


    .
    ├── ...
    └── data                    
        ├── ...         
        └── datasets                
            ├── ...
            └── dataset1
           		├── train
           		│   ├── img1.jpg    
           		│   ├── img1.xml    # Annotation for img1 in VOC Pascal format
           		│   └── ...
           		└── test
           		    ├── img2.jpg
           		    ├── img2.xml    # Annotation for img2 in VOC Pascal format
           		    └── ...


The dataset have to be split into a train and test directory, this can be done using split_dataset.py in resources.


## Authors

* **Simen Viken Grini** - *Initial work* - [simenvg](https://github.com/simenvg)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

