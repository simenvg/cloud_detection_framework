import xml.etree.ElementTree as ET
import os
import sys
import random


if len(sys.argv) <= 1:
    print("Argument = Complete path to image folder")
    sys.exit()

path = sys.argv[1]


def get_classes(path):
    classes = []
    for filename in os.listdir(path):
        if filename.endswith(".xml"):
            tree = ET.parse(os.path.join(path, filename))
            root = tree.getroot()
            for child in root:
                for child in child:
                    if child.tag == "name":
                        if child.text not in classes:
                            classes.append(child.text)
        else:
            continue
    return classes


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(filename, classes):
    xml_annotation_file = filename[:-4] + '.xml'
    txt_annotation_file = filename[:-4] + '.txt'
    in_file = open(os.path.join(path, xml_annotation_file), 'r')
    out_file = open(os.path.join(path, txt_annotation_file), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')


def generate_yolo_files(path):
    classes = get_classes(path)
    train_file = open('train.txt', 'w')
    test_file = open('test.txt', 'w')
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            if random.random() >= 0.50:
                train_file.write(os.path.join(path, filename) + '\n')
            else:
                test_file.write(os.path.join(path, filename) + '\n')
            convert_annotation(filename, classes)
        else:
            continue
    train_file.close()
    test_file.close()


if __name__ == '__main__':
    generate_yolo_files(path)
