import sys
import os
import argparse
import sqlite3 as db

parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('DATA_PATH', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
parser.add_argument('DARKNET_PATH', type=str, nargs=1,
                    help='Path to darknet folder')

args = parser.parse_args()
DATA_PATH = args.DATA_PATH[0]
DARKNET_PATH = args.DARKNET_PATH[0]
# Adds Darknet to python path
sys.path.append(os.path.join(DARKNET_PATH, 'python/'))
import darknet as dn


def find_best_weights_file(data_path):
    files = os.listdir(os.path.join(data_path, 'model', 'backup'))
    highest_iter = 0
    highest_iter_path = ''
    for file in files:
        if file.endswith('.weights'):
            filename_split = file.split('_')
            num = int(filename_split[-1].strip()[:-8])
            print(num)
            if num > highest_iter:
                highest_iter_path = os.path.join(data_path, 'model', 'backup', file)
    print('Weights PATH: ', highest_iter_path)
    return highest_iter_path


# Initialize detector
dn.set_gpu(0)
net = dn.load_net(os.path.join(DATA_PATH, "model",
                               "yolo-obj_test.cfg"), find_best_weights_file(DATA_PATH), 0)
meta_data_net = dn.load_meta(os.path.join(DATA_PATH, "model", "obj.data"))


class Box(object):
    """docstring for Box"""

    def __init__(self, class_name, xmin, xmax, ymin, ymax, confidence=None):
        self.class_name = class_name
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.confidence = confidence


def initialize_database():
    conn = db.connect(os.path.join(DATA_PATH, 'results', 'detections.db'))
    c = conn.cursor()
    c.execute('''CREATE TABLE detections
                         (image_name text, xmin integer, xmax integer, ymin integer, ymax integer, class_name text, confidence real)''')
    return conn


def add_to_db(conn, image_name, xmin, xmax, ymin, ymax, class_name, confidence):
    c = conn.cursor()
    c.execute("INSERT INTO detections (image_name, xmin, xmax, ymin, ymax, class_name, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)"), (
        image_name, xmin, xmax, ymin, ymax, class_name, confidence)


def convert_yolo_format(x_center, y_center, width, height):
    x_min = float(x_center) - float(width) / 2
    x_max = float(x_center) + float(width) / 2
    y_min = float(y_center) - float(height) / 2
    y_max = float(y_center) + float(height) / 2
    return [x_min, x_max, y_min, y_max]


def get_detected_boxes(yolo_output):
    boxes = []
    for detection in yolo_output:
        coordinates = convert_yolo_format(
            detection[2][0], detection[2][1], detection[2][2], detection[2][3])
        boxes.append(Box(detection[0], coordinates[0],
                         coordinates[1], coordinates[2], coordinates[3], confidence=detection[1]))
    return boxes


def get_yolo_detections(image_name, net, meta_data_net, thresh=0.5):
    detections = dn.detect(net, meta_data_net, os.path.join(
        DARKNET_PATH, image_name.strip()), thresh=thresh)
    print(detections)
    return get_detected_boxes(detections)


def write_detections_to_db(image_filepaths, thresh=0.5):
    conn = initialize_database()
    for image in image_filepaths:
        boxes = get_yolo_detections(image, net, meta_data_net, thresh)
        for box in boxes:
            add_to_db(conn, image, box.xmin, box.xmax, box.ymin,
                      box.ymax, box.class_name, box.confidence)
    conn.commit()
    conn.close()


def set_test_datasets(data_path):
    datasets = os.listdir(os.path.join(data_path, 'datasets'))
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
    test_image_filepaths = []
    for training_dataset_path in training_dataset_paths:
        for file in os.listdir(os.path.join(training_dataset_path, 'test')):
            if file.endswith('.jpg'):
                test_image_filepaths.append(os.path.join(
                    training_dataset_path, 'test', file))
    return test_image_filepaths


if __name__ == '__main__':
    result_path = os.path.join(DATA_PATH, 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    write_detections_to_db(set_test_datasets(DATA_PATH), thresh=0.05)
