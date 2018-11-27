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


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


for file_name in file_names:
    file_path = os.path.join(DATA_PATH, file_name)
    prec_recall_dict = load_pickle(file_path)
    name = prec_recall_dict['name']
    precisions = prec_recall_dict['precisions']
    recalls = prec_recall_dict['recalls']
    plt.plot(recalls, precisions, label=name)
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

plt.show()
