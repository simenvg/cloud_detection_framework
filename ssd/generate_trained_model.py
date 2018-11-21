# import sys
import os
import argparse
# import sqlite3 as db
import subprocess


def find_highest_model_file(data_path):
    highest_iteration = 0
    for file in os.listdir(os.path.join(data_path, 'models', 'model')):
        if file.endswith('.meta'):
            file_name_split = file.split('.')
            iteration = int(file_name_split[1][5:])
            if iteration > highest_iteration:
                highest_iteration = iteration
    return highest_iteration


def main(data_path):
    highest_iteration = str(find_highest_model_file(data_path))
    pipeline_config_path = os.path.join(data_path, 'SSD_mobilenet',
                                        'models', 'model', 'ssd_mobilenet_v1_coco_edited.config')
    home_path = data_path.replace('/data', '')
    subprocess.call(['python', home_path + '/models/research/object_detection/export_inference_graph.py',
                     '--input_type=image_tensor',
                     '--pipeline_config_path=' + pipeline_config_path,
                     '--trained_checkpoint_prefix=' + data_path + '/models/model/model.ckpt-' + highest_iteration,
                     '--output_directory=' + data_path + '/trained_ssd_model'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input path to darknet')
    parser.add_argument('DATA_PATH', type=str, nargs=1,
                        help='Set path to data folder, containg datasets')

    args = parser.parse_args()
    DATA_PATH = args.DATA_PATH[0]
    main(DATA_PATH)
