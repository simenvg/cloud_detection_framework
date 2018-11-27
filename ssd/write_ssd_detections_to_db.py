# # Imports


import numpy as np
import os
import sys
import tensorflow as tf
import argparse
import sqlite3 as db

from distutils.version import StrictVersion
from PIL import Image


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_multiple_images(image_paths, graph):
    with graph.as_default():
        with tf.Session() as sess:
            output_dicts = []
            for image_path in image_paths:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)

                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                               real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                               real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image_np, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                width, height = image.size
                output_dict['width'] = width
                output_dict['height'] = height
                output_dict['name'] = image_path
                output_dicts.append(output_dict)
    return output_dicts


def set_test_datasets(data_path):
    datasets = os.listdir(os.path.join(data_path, 'datasets'))
    if len(datasets) == 0:
        print('No datasets in ~/data, run config_new_dataset.py on your dataset and move the dataset folder to ~/data')
    for i in range(0, len(datasets)):
        print('[', i, ']', datasets[i])
    user_input = str(input(
        'Input the number for the datasets you wish to test on, separate numbers with space: ')).split()
    training_dataset_paths = []
    for dataset_index in user_input:
        training_dataset_paths.append(
            os.path.join(data_path, 'datasets', datasets[int(dataset_index)]))
    test_image_filepaths = []
    for training_dataset_path in training_dataset_paths:
        for file in os.listdir(os.path.join(training_dataset_path, 'test')):
            if file.endswith('.jpg'):
                test_image_filepaths.append(os.path.join(
                    training_dataset_path, 'test', file))
    new_test_file = open(os.path.join(data_path, 'model', 'test.txt'), 'w')
    for test_image_filepath in test_image_filepaths:
        new_test_file.write(test_image_filepath + '\n')
    new_test_file.close()
    return test_image_filepaths


def initialize_database():
    conn = db.connect(os.path.join(DATA_PATH, 'results', 'detections.db'))
    c = conn.cursor()
    c.execute('''CREATE TABLE detections
                         (image_name text, xmin integer, xmax integer, ymin integer, ymax integer, class_name text, confidence real)''')
    return conn


def add_to_db(conn, image_name, xmin, xmax, ymin, ymax, class_name, confidence):
    c = conn.cursor()
    c.execute("INSERT INTO detections (image_name, xmin, xmax, ymin, ymax, class_name, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)", (
        image_name, xmin, xmax, ymin, ymax, str(class_name), float(confidence)))


def main(data_path):
    conn = initialize_database()
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = os.path.join(
        data_path, 'trained_ssd_model', 'frozen_inference_graph.pb')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    test_image_paths = set_test_datasets(data_path)
    output_dicts = run_inference_for_multiple_images(test_image_paths, detection_graph)
    for output_dict in output_dicts:
        detection_classes = output_dict['detection_classes']
        detection_scores = output_dict['detection_scores']
        detection_boxes = output_dict['detection_boxes']
        print(len(detection_scores))
        width = output_dict['width']
        height = output_dict['height']
        for i in range(len(detection_boxes)):
            xmin = int(detection_boxes[i][1] * width)
            xmax = int(detection_boxes[i][3] * width)
            ymin = int(detection_boxes[i][0] * height)
            ymax = int(detection_boxes[i][2] * height)

            add_to_db(conn, output_dict['name'], xmin, xmax, ymin,
                      ymax, detection_classes[i], detection_scores[i])
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input path to darknet')
    parser.add_argument('DATA_PATH', type=str, nargs=1,
                        help='Set path to data folder, containg datasets')
    args = parser.parse_args()
    DATA_PATH = args.DATA_PATH[0]
    home_path = DATA_PATH.replace('/data', '')
    sys.path.append(os.path.join(home_path, 'models',
                                 'research', 'object_detection'))
    from object_detection.utils import ops as utils_ops

    if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
        raise ImportError(
            'Please upgrade your TensorFlow installation to v1.9.* or later!')
    # from utils import label_map_util

    # from utils import visualization_utils as vis_util

    main(DATA_PATH)
