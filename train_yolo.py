import argparse
import os
from config_new_dataset import generate_yolo_files
import shutil


parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('DARKNET_PATH', type=str, nargs=1,
                    help='Set path to darknet')
parser.add_argument('DATA_PATH', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')

args = parser.parse_args()
DARKNET_PATH = args.DARKNET_PATH[0]
DATA_PATH = args.DATA_PATH[0]


def set_training_datasets():
    try:
        datasets = os.listdir(DATA_PATH)
    except Exception as e:
        print('No folder named ~/data')
        print('Exception: ', e)
    if len(datasets) == 0:
        print('No datasets in ~/data, run config_new_dataset.py on your dataset and move the dataset folder to ~/data')

    for i in range(0, len(datasets)):
        print('[', i, ']   ', datasets[i])
    user_input = str(raw_input(
        'Input the number for the datasets you wish to train on, separate numbers with space: ')).split()
    print(user_input)
    training_dataset_paths = []
    for dataset_index in user_input:
        training_dataset_paths.append(
            os.path.join(DATA_PATH, datasets[int(dataset_index)]))
    return training_dataset_paths


def generate_yolo_train_files():
    training_dataset_paths = set_training_datasets()
    training_folder = os.path.join(DATA_PATH, 'tmp')
    if os.path.exists(training_folder):
        shutil.rmtree(training_folder, ignore_errors=True)
    os.makedirs(training_folder)
    for training_dataset in training_dataset_paths:
        for filename in os.listdir(training_dataset):
            shutil.copy2(os.path.join(training_dataset, filename), os.path.join(DATA_PATH, 'tmp'))
    generate_yolo_files(training_folder)
    # CFG
    # OBJ.Names obj.data in darknet/data
    # Set Batch and subdivisions in CFG
    # Set classes in CFG, line 620, 696, 783
    # set Filters (classes + 5) * 3 in CFG, line 602, 689, 776
    # weights saved every 1000 iteration

if __name__ == '__main__':
    generate_yolo_train_files()