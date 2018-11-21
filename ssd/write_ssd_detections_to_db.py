# # Imports


import numpy as np
import os
import sys
import tensorflow as tf
import argparse

from distutils.version import StrictVersion
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v1.9.* or later!')


from utils import label_map_util

from utils import visualization_utils as vis_util



def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
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
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# In[ ]:

def set_test_datasets(data_path):
    datasets = os.listdir(os.path.join(data_path, 'datasets'))
    if len(datasets) == 0:
        print 'No datasets in ~/data, run config_new_dataset.py on your dataset and move the dataset folder to ~/data'
    for i in range(0, len(datasets)):
        print '[', i, ']', datasets[i]
    user_input = str(raw_input(
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
    return test_image_filepaths


def main(data_path, detection_graph):
    # MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = os.path.join(data_path, 'trained_ssd_model', 'frozen_inference_graph.pb')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(data_path, 'mscoco_label_map.pbtxt')

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    #category_index = label_map_util.create_category_index_from_labelmap(
     #   PATH_TO_LABELS, use_display_name=True)

    test_image_paths = set_test_datasets(data_path)
    for image_path in test_image_paths:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        print(output_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input path to darknet')
    parser.add_argument('DATA_PATH', type=str, nargs=1,
                        help='Set path to data folder, containg datasets')
    args = parser.parse_args()
    DATA_PATH = args.DATA_PATH[0]
    main(DATA_PATH)
