# Cloud Detection Framework

This project is a part of my master thesis and serves as a framework for using Googles cloud computing engine to easily train and test object detectors on custom data.

## Getting Started

### Prerequisites


 * Setup a Google cloud instance with GPU and Ubuntu 16.04. This is tested on a Nvidia Tesla K80 GPU. [See Googles get started guide](https://cloud.google.com/compute/docs/instances/create-start-instance)
 * Download and install the [gcloud SDK and command line tool](https://cloud.google.com/sdk/install)


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

To begin training we need some data. The data must be labelled in the VOC Pascal format, the xml files and the jpg files should be in the same folder. To label images [LabelImg](https://github.com/tzutalin/labelImg) which will label the images in the correct format. 

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

To upload a dataset to the Google Cloud instance use the following command

```
gcloud compute scp --recurse /LOCAL/PATH/TO/DATASET <instance-name>:~/data/datasets --zone <zone>
```

## Training

You can now run train_yolo.py which will ask the user which datasets to use for training.

```
python cloud_detection_framework/yolo/train_yolo.py /FULL/PATH/TO/DARKNET /FULL/PATH/TO/DATA/DIRECTORY
```

YOLO is implemented in Python 2, and SSD in Python 3, not to make it harder, but it was the only way while still using the original implementation I found. This was solved using a virtual environment for everything other than YOLO related stuff. Activate by using the command

```
source venv/bin/activate
```
from the home directory. When the Python 3 virtual environment is activated the following command can be run to train ssd
```
python cloud_detection_framework/ssd/train_ssd.py /FULL/PATH/TO/DATA/DIRECTORY
```

## Testing


## Authors

* **Simen Viken Grini** - [simenvg](https://github.com/simenvg)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

