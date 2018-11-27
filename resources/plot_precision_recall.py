import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('DATA_PATH', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
args = parser.parse_args()
DATA_PATH = args.DATA_PATH[0]

file_names = os.listdir(DATA_PATH)


def load_file(file_path):
    file = open(file_path, 'r')
    prec_recall_dict = {}
    prec_recall_dict['name'] = file.readline()
    prec_recall_dict['precisions'] = file.readline().split(' ')
    prec_recall_dict['recalls'] = file.readline().split(' ')
    file.close()
    return prec_recall_dict


for file_name in file_names:
    file_path = os.path.join(DATA_PATH, file_name)
    prec_recall_dict = load_file(file_path)
    name = prec_recall_dict['name']
    precisions = prec_recall_dict['precisions']
    recalls = prec_recall_dict['recalls']
    plt.plot(recalls, precisions, label=name)
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.title('Precision/Recall curve')

plt.show()
