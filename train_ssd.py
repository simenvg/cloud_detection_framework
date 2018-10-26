import subprocess
import os
import argparse


parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('DATA_PATH', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')

args = parser.parse_args()
DATA_PATH = args.DATA_PATH[0]


def edit_config_file(data_path):
    config_file = open(os.path.join(data_path, 'SSD_mobilenet',
                                    'models', 'model', 'ssd_mobilenet_v1_coco.config'), 'r')
    new_config_file = open(os.path.join(data_path, 'SSD_mobilenet',
                                        'models', 'model', 'ssd_mobilenet_v1_coco_edited.config'), 'w')
    lines = config_file.readlines()
    num_classes = get_num_classes(
        os.path.join(data_path, 'tmp', 'classes.txt'))
    lines[8] = '    num_classes: ' + str(num_classes) + '\n'
    lines[155] = '  fine_tune_checkpoint: "' + data_path + \
        '/SSD_mobilenet/models/model/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"\n'
    lines[174] = '    input_path: "' + data_path + \
        '/SSD_mobilenet/data/train.record"\n'
    lines[176] = '  label_map_path: "' + data_path + \
        '/SSD_mobilenet/data/label_map.pbtxt"\n'
    lines[188] = '    input_path: "' + data_path + \
        '/SSD_mobilenet/data/test.record"\n'
    lines[190] = '  label_map_path: "' + data_path + \
        '/SSD_mobilenet/data/label_map.pbtxt"\n'
    for line in lines:
        new_config_file.write(line)
    config_file.close()
    new_config_file.close()


def get_num_classes(class_file_path):
    class_file = open(class_file_path, 'r')
    lines = class_file.readlines()
    return len(lines)


def start_training(data_path):
    pipeline_config_path = os.path.join(data_path, 'SSD_mobilenet',
                                        'models', 'model', 'ssd_mobilenet_v1_coco_edited.config')
    subprocess.call(['python', '~/models/research/object_detection/model_main.py',
                     'pipeline_config_path=' + pipeline_config_path,
                     '--model_dir=' + data_path + '/models/model',
                     '--num_train_steps=50000',
                     '--sample_1_of_n_eval_examples=1',
                     '--alsologtostderr'])


def main(data_path):
    edit_config_file(data_path)
    start_training(data_path)


if __name__ == '__main__':
    main(DATA_PATH)
