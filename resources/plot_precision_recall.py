import pickle
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('DATA_PATH', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
args = parser.parse_args()
DATA_PATH = args.DATA_PATH[0]

file_names = os.listdir(DATA_PATH)

for file_name in file_names:
    file = open(os.path.join(DATA_PATH, file_name), 'r')
    prec_recall_dict = pickle.loads(file.read())
    name = prec_recall_dict['name']
    precisions = prec_recall_dict['precisions']
    recalls = prec_recall_dict['recalls']
    plt.plot(recalls, precisions, label=name)
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

plt.show()
