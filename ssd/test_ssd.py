# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import xml.etree.ElementTree as ET
import os
import cv2
import sqlite3 as db
import argparse
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


class Box(object):
    """docstring for Box"""

    def __init__(self, cls, x_min, x_max, y_min, y_max, confidence=None):
        self.cls = cls
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.confidence = confidence


def get_GT_boxes(label_filepath):
    in_file = open(os.path.join(label_filepath), 'r')
    tree = ET.parse(in_file)
    root = tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        if obj.find('name').text == 'boat':
            boxes.append(Box(obj.find('name').text, float(xmlbox.find('xmin').text), float(
                xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)))
    return boxes


def get_intersected_area(box1, box2):
    dx = min(box1.x_max, box2.x_max) - max(box1.x_min, box2.x_min)
    dy = min(box1.y_max, box2.y_max) - max(box1.y_min, box2.y_min)
    if dy <= 0 or dx <= 0:
        return -1
    else:
        return dx * dy


def get_iou(box1, box2):
    area_box1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
    area_box2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
    intersected_area = get_intersected_area(box1, box2)
    # print(intersected_area)
    if intersected_area == -1:
        return -1
    else:
        return intersected_area / (area_box1 + area_box2 - intersected_area)


def valid_detection(detected_box, gt_box, iou_thresh=0.5):
    return get_iou(detected_box, gt_box) >= iou_thresh


def get_detections_stats(conn, data_path, iou_thresh, confidence_thresh=0.25):
    true_positives = 0
    num_detections = 0
    num_gt_boxes = 0
    c = conn.cursor()
    test_file = open(os.path.join(data_path, 'model', 'test.txt'), 'r')
    image_filepaths = test_file.readlines()
    test_file.close()
    for img in image_filepaths:
        gt_boxes = get_GT_boxes(os.path.join((img.strip()[:-4] + '.xml')))
        c.execute('SELECT * FROM detections WHERE image_name=? and confidence>=? and class_name=?',
                  (img.strip(), confidence_thresh, 'boat'))
        detections = c.fetchall()
        num_detections += len(detections)
        for gt_box in gt_boxes:
            for i in range(len(detections) - 1, -1, -1):
                detected_box = Box(detections[i][5], detections[i][1], detections[i]
                                   [2], detections[i][3], detections[i][4], detections[i][6])
                if detected_box.confidence >= confidence_thresh:
                    if valid_detection(detected_box, gt_box, iou_thresh=iou_thresh):
                        true_positives += 1
                        detections.remove(detections[i])
                        break
        num_gt_boxes += len(gt_boxes)
    print('TP: ', true_positives, '   num_detections: ',
          num_detections, '   num_gt: ', num_gt_boxes)
    return [true_positives, num_detections, num_gt_boxes]


def get_precision_recall(conn, data_path, iou_thresh, confidence_thresh=0.25):
    [true_positives, num_detections, num_gt_boxes] = get_detections_stats(conn, data_path, iou_thresh, confidence_thresh)
    precision = 0
    if num_detections > 0:
        precision = float(true_positives) / float(num_detections)
    recall = 0
    if num_gt_boxes > 0:
        recall = float(true_positives) / float(num_gt_boxes)
    return (precision, recall)


def get_confusion_matrix(conn, data_path, iou_thresh, confidence_thresh=0.25):
    [true_positives, num_detections, num_gt_boxes] = get_detections_stats(conn, data_path, iou_thresh, confidence_thresh)
    false_positives = num_detections - true_positives
    false_negatives = num_gt_boxes - true_positives
    file = open(os.path.join(data_path, 'results', 'confusion_matrix.txt'), 'w')
    file.write('true_positives = ' + str(true_positives))
    file.write('false_positives = ' + str(false_positives))
    file.write('false_negatives = ' + str(false_negatives))
    file.write('num_detections = ' + str(num_detections))
    file.write('num_gt_boxes = ' + str(num_gt_boxes))
    file.close()
    return [true_positives, false_positives, false_negatives]


def save_images_with_boxes(conn, data_path, conf_thresh=0.25):
    c = conn.cursor()
    test_file = open(os.path.join(data_path, 'model', 'test.txt'), 'r')
    image_filepaths = test_file.readlines()
    test_file.close()
    i = 0
    for img in image_filepaths:
        img_name = img.strip().split('/')[-1]
        gt_boxes = get_GT_boxes(os.path.join(
            '', (img.strip()[:-4] + '.xml')))
        c.execute('SELECT * FROM detections WHERE image_name=? AND confidence>=?',
                  (img.strip(), conf_thresh))
        detections = c.fetchall()
        image = cv2.imread(img.strip())
        print(img.strip())
        if image is None:
            print('No image')
            exit()
        for box in gt_boxes:
            cv2.rectangle(image, (int(box.x_min), int(box.y_max)),
                          (int(box.x_max), int(box.y_min)), GREEN, 6)
        for box in detections:
            if (box[5] == 'building'):
                color = BLUE
            else:
                color = RED
            cv2.rectangle(image, (int(box[1]), int(box[3])),
                          (int(box[2]), int(box[4])), color, 6)
        cv2.imwrite(os.path.join(data_path, 'results',
                                 img_name), image)
        i += 1


def write_prec_recall_to_file(data_path, precisions, recalls, name='SSD'):
    file = open(os.path.join(data_path, 'results', 'prec_recalls.txt'), 'w')
    file.write(name + '\n')
    for i in range(len(precisions)):
        file.write(str(precisions[i]))
        if i != len(precisions):
            file.write(' ')
    file.write('\n')
    for i in range len(recalls):
        file.write(str(recalls[i]))
        if i != len(recalls):
            file.write(' ')
    file.write('\n')
    file.close()


def main(data_path):
    conn = db.connect(os.path.join(data_path, 'results', 'detections.db'))
    save_images_with_boxes(conn, data_path)
    conf_threshs = [x * 0.01 for x in range(0, 100)]
    precisions = []
    recalls = []
    for conf_thresh in conf_threshs:
        (precision, recall) = get_precision_recall(
            conn, data_path, 0.5, conf_thresh)
        precisions.append(precision)
        recalls.append(recall)
    print(precisions)
    print(recalls)
    write_prec_recall_to_file(data_path, precisions, recalls)
    get_confusion_matrix(conn, data_path, 0.5)
    # print(get_precision_recall(conn, data_path, 0.5))
    plt.plot(recalls, precisions)
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(data_path, 'results', 'prec_recall.png'))
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input path to darknet')
    parser.add_argument('DATA_PATH', type=str, nargs=1,
                        help='Set path to data folder, containg datasets')
    args = parser.parse_args()
    DATA_PATH = args.DATA_PATH[0]
    main(DATA_PATH)
