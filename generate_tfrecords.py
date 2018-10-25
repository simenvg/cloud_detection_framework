"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('data_path', '', 'Path to data folder')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def get_class_dict(classes_path):
    class_file = open(classes_path, 'r')
    lines = class_file.readlines()
    classes = {}
    for i in range(0, len(lines)):
        classes[lines[i].strip()] = i + 1
    return classes


def class_mapper(class_name):
    if class_name == 'motor_vessel':
        return 1
    elif class_name == 'kayak':
        return 2
    elif class_name == 'sailboat_motor':
        return 3
    elif class_name == 'sailboat_sail':
        return 4
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, class_map):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        # classes.append(class_map[row['class']])
        classes.append(class_mapper(row['class']))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    folders = ['train', 'test']
    class_map = get_class_dict(os.path.join(
        FLAGS.data_path, 'tmp', 'classes.txt'))
    for folder in folders:
        writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_path, 'tmp', folder + '.record'))
        image_path = os.path.join(FLAGS.data_path, 'tmp', folder, 'JPEGImages')
        examples = pd.read_csv(os.path.join(
            FLAGS.data_path, 'tmp', folder + '.csv'))
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, image_path, class_map)
            writer.write(tf_example.SerializeToString())
        writer.close()
    output_path = os.path.join(FLAGS.data_path, 'tmp')
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
