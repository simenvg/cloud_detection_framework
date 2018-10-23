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
        datasets = os.listdir(data_path)
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
            os.path.join(data_path, datasets[int(dataset_index)]))
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


def setup_train_data():
    datasets = set_training_datasets()
    [tmp_folder_path, tmp_folder_train_path,
        tmp_folder_test_path] = setup_tmp_folder()
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
    classes = ['none']
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
    class_file = open(os.path.join(path, 'classes.txt'), 'w')
    for clas in classes:
        if clas == classes[-1]:
            class_file.write(clas)
        else:
            class_file.write(clas + '\n')
    class_file.close()


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    classes = setup_train_data()
    generate_classes_file(os.path.join(DATA_PATH, 'tmp'), classes)
    folders = ['train', 'test']
    for folder in folders:
        # image_path = os.path.join(os.getcwd(), 'annotations')
        xml_df = xml_to_csv(os.path.join(DATA_PATH, 'tmp', folder, 'Annotations'))
        xml_df.to_csv(folder + '.csv', index=None)
    print('Successfully converted xml to csv.')


main()
