import argparse
import os
import convert_to_yolo_format
import shutil
import subprocess


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
        datasets = os.listdir(os.path.join(DATA_PATH, 'datasets'))
    except Exception as e:
        print 'No folder named ~/data'
        print 'Exception: ', e
        exit()
    if len(datasets) == 0:
        print 'No datasets in ~/data, run config_new_dataset.py on your dataset and move the dataset folder to ~/data'

    for i in range(0, len(datasets)):
        print '[', i, ']', datasets[i]
    user_input = str(raw_input(
        'Input the number for the datasets you wish to train on, separate numbers with space: ')).split()
    training_dataset_paths = []
    for dataset_index in user_input:
        training_dataset_paths.append(
            os.path.join(DATA_PATH, 'datasets', datasets[int(dataset_index)]))
    return training_dataset_paths


def setup_model_folder():
    model_folder_path = os.path.join(DATA_PATH, 'model')
    if os.path.exists(model_folder_path):
        shutil.rmtree(model_folder_path, ignore_errors=True)
    os.makedirs(model_folder_path)
    os.makedirs(os.path.join(model_folder_path, 'backup'))
    return model_folder_path


def generate_yolo_train_files():
    test_and_train_paths = []
    training_dataset_paths = set_training_datasets()
    model_folder_path = setup_model_folder()
    train_txt = open(os.path.join(model_folder_path, 'train.txt'), 'w')
    test_txt = open(os.path.join(model_folder_path, 'test.txt'), 'w')
    for training_dataset_path in training_dataset_paths:
        test_and_train_paths.extend([os.path.join(training_dataset_path, 'train'), os.path.join(training_dataset_path, 'test')])
        for filename in os.listdir(os.path.join(training_dataset_path, 'train')):
            if filename.endswith('.jpg'):
                train_txt.write(os.path.join(
                    training_dataset_path, 'train', filename) + '\n')
        for filename in os.listdir(os.path.join(training_dataset_path, 'test')):
            if filename.endswith('.jpg'):
                test_txt.write(os.path.join(
                    training_dataset_path, 'test', filename) + '\n')
    test_txt.close()
    train_txt.close()
    classes = convert_to_yolo_format.get_classes(test_and_train_paths)
    for path in test_and_train_paths:
        convert_to_yolo_format.generate_yolo_annotation_files(path, classes)
    convert_to_yolo_format.generate_classes_file(model_folder_path, classes)

    return len(classes)


def update_cfg_file(num_classes):
    num_filters = (num_classes + 5) * 3
    cfg_file = open(os.path.join(
        DARKNET_PATH, 'cfg', 'yolov3.cfg'), 'r')
    lines = cfg_file.readlines()
    cfg_file.close()
    class_lines = [609, 695, 782]
    filter_lines = [602, 688, 775]
    for class_line in class_lines:
        lines[class_line] = 'classes=' + str(num_classes) + '\n'
    for filter_line in filter_lines:
        lines[filter_line] = 'filters=' + str(num_filters) + '\n'
    new_cfg_file = open(os.path.join(
        DATA_PATH, 'model', 'yolo-obj_test.cfg'), 'w')
    for line in lines:
        new_cfg_file.write(line)
    new_cfg_file.close()


def generate_obj_data(num_classes):
    line1 = 'classes = ' + str(num_classes) + '\n'
    line2 = 'train = ' + os.path.join(DATA_PATH, 'model', 'train.txt') + '\n'
    line3 = 'valid = ' + os.path.join(DATA_PATH, 'model', 'test.txt') + '\n'
    line4 = 'names = ' + os.path.join(DATA_PATH, 'model', 'classes.names') + '\n'
    line5 = 'backup = ' + os.path.join(DATA_PATH, 'model', 'backup/')
    lines = [line1, line2, line3, line4, line5]
    obj_data_file = open(os.path.join(DATA_PATH, 'model', 'obj.data'), 'w')
    for line in lines:
        obj_data_file.write(line)
    obj_data_file.close()


def train_yolo():
    subprocess.call(['/' + DARKNET_PATH + '/darknet', 'detector', 'train',
                     DATA_PATH + '/model/obj.data', DATA_PATH + '/model/yolo-obj_test.cfg', DARKNET_PATH + '/darknet53.conv.74'])


if __name__ == '__main__':
    num_classes = generate_yolo_train_files()
    update_cfg_file(num_classes)
    generate_obj_data(num_classes)
    train_yolo()
