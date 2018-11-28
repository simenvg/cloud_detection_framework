import xml.etree.ElementTree as ET
import argparse
import os

parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('DATA_PATH', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
args = parser.parse_args()
DATA_PATH = args.DATA_PATH[0]

files = os.listdir(DATA_PATH)

for file in files:
    if file.endswith('.xml'):
        tree = ET.parse(os.path.join(DATA_PATH, file))
        root = tree.getroot()
        for elem in tree.iterfind('folder'):
            print(elem.text)
            elem.text = os.path.join('train')
            print(elem.text)
        tree.write(os.path.join(DATA_PATH, file))
