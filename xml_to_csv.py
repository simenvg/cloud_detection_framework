import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import shutil
import argparse

parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('DATA_PATH', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')

args = parser.parse_args()
DATA_PATH = args.DATA_PATH[0]


def set_training_datasets(data_path):
    try:
        datasets = os.listdir(os.path.join(data_path, 'datasets'))
    except Exception as e:
        print('No folder named ~/data')
        print('Exception: ', e)
        exit()
    if len(datasets) == 0:
        print('No datasets in ~/data, run config_new_dataset.py on your dataset and move the dataset folder to ~/data')
    for i in range(0, len(datasets)):
        print('[', i, ']', datasets[i])
    user_input = str(input(
        'Input the number for the datasets you wish to train on, separate numbers with space: ')).split()
    training_dataset_paths = []
    for dataset_index in user_input:
        training_dataset_paths.append(
            os.path.join(data_path, 'datasets', datasets[int(dataset_index)]))
    return training_dataset_paths


def setup_tmp_folder(data_path):
    tmp_folder_path = os.path.join(data_path, 'tmp')
    tmp_folder_test_path = os.path.join(tmp_folder_path, 'test')
    tmp_folder_train_path = os.path.join(tmp_folder_path, 'train')
    train_and_test = [tmp_folder_test_path, tmp_folder_train_path]
    if os.path.exists(tmp_folder_path):
        shutil.rmtree(tmp_folder_path, ignore_errors=True)
    os.makedirs(tmp_folder_path)
    os.makedirs(tmp_folder_test_path)
    os.makedirs(tmp_folder_train_path)
    for directory in train_and_test:
        os.makedirs(os.path.join(directory, 'Annotations'))
        os.makedirs(os.path.join(directory, 'JPEGImages'))
    return [tmp_folder_path, tmp_folder_train_path, tmp_folder_test_path]


def setup_train_data(data_path):
    datasets = set_training_datasets(data_path)
    [tmp_folder_path, tmp_folder_train_path,
        tmp_folder_test_path] = setup_tmp_folder(data_path)
    for dataset in datasets:
        for filename in os.listdir(os.path.join(dataset, 'train')):
            if filename.endswith('.jpg'):
                shutil.copy2(os.path.join(dataset, 'train', filename),
                             os.path.join(tmp_folder_train_path, 'JPEGImages'))
            elif filename.endswith('.xml'):
                shutil.copy2(os.path.join(dataset, 'train', filename),
                             os.path.join(tmp_folder_train_path, 'Annotations'))
        for filename in os.listdir(os.path.join(dataset, 'test')):
            if filename.endswith('.jpg'):
                shutil.copy2(os.path.join(dataset, 'test', filename),
                             os.path.join(tmp_folder_test_path, 'JPEGImages'))
            elif filename.endswith('.xml'):
                shutil.copy2(os.path.join(dataset, 'test', filename),
                             os.path.join(tmp_folder_test_path, 'Annotations'))
    return get_classes([os.path.join(tmp_folder_test_path, 'Annotations'), os.path.join(tmp_folder_train_path, 'Annotations')])


def get_classes(paths):
    classes = []
    for path in paths:
        for filename in os.listdir(path):
            if filename.endswith(".xml"):
                tree = ET.parse(os.path.join(path, filename))
                root = tree.getroot()
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    if label not in classes:
                        classes.append(label)
            else:
                continue
    return classes


def generate_classes_file(path, classes):
    class_file_txt = open(os.path.join(path, 'tmp', 'classes.txt'), 'w')
    class_file_pbtxt = open(os.path.join(path, 'SSD_mobilenet', 'data', 'label_map.pbtxt'), 'w')
    for i in range(0, len(classes)):
        if classes[i] == classes[-1]:
            class_file_txt.write(classes[i])
        else:
            class_file_txt.write(classes[i] + '\n')
        class_file_pbtxt.write('item { \n  id: ' + str(i + 1) + '\n  name: \'' + classes[i] + '\' \n } \n \n')
    class_file_txt.close()


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            xmlbox = member.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymin = int(xmlbox.find('ymin').text)
            ymax = int(xmlbox.find('ymax').text)
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    classes = setup_train_data(DATA_PATH)
    generate_classes_file(DATA_PATH, classes)
    folders = ['train', 'test']
    for folder in folders:
        # image_path = os.path.join(os.getcwd(), 'annotations')
        xml_df = xml_to_csv(os.path.join(
            DATA_PATH, 'tmp', folder, 'Annotations'))
        xml_df.to_csv(os.path.join(DATA_PATH, 'tmp', folder + '.csv'), index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()
