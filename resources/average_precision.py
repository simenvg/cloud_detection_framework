import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('results_path', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
args = parser.parse_args()
results_path = args.results_path[0]


def get_precision_at_r(precisions, recalls, recall_value):
    indexes = []
    for i in range(len(recalls)):
        if float(recalls[i]) >= recall_value:
            indexes.append(i)
    print(indexes)
    if len(indexes) == 0:
        return 0
    highest_prec = float(precisions[indexes[0]])
    for index in indexes:
        if float(precisions[index]) > float(highest_prec):
            highest_prec = float(precisions[index])
    print(highest_prec)
    return highest_prec


def get_ap(precisions, recalls):
    recall_levels = [x * 0.1 for x in range(0, 10)]
    recall_levels.append(1)
    p = 0
    for recall_level in recall_levels:
        p += get_precision_at_r(precisions, recalls, recall_level)
    return p / len(recall_levels)


def load_file(file_path):
    file = open(file_path, 'r')
    prec_recall_dict = {}
    prec_recall_dict['name'] = file.readline()
    prec_recall_dict['precisions'] = file.readline().split(' ')
    prec_recall_dict['recalls'] = file.readline().split(' ')
    file.close()
    return prec_recall_dict


def get_files2(results_path):
    file = open('ap.txt', 'w')
    trained_on = ['trained_on_trf_bc_bf',
                  'trained_on_trf_bc_bf_bb_build', 'trained_on_trf_bc_bf_bbnb']
    tested_on = ['tested_on_b_and_b', 'tested_on_trf',
                 'tested_on_bc_bf', 'tested_on_bc_bf_trf']
    ssd_yolo = ['results-yolo', 'results-ssd']
    for train in trained_on:
        for test in tested_on:
            for det in ssd_yolo:
                prec_recall_dict = load_file(os.path.join(results_path, train, test, det, 'prec_recalls.txt'))
                ap = get_ap(prec_recall_dict['precisions'], prec_recall_dict['recalls'])
                file.write(train + '  ' + test + '  ' + det + '   AP:' + str(ap) + '\n')
    file.close()

get_files2(results_path)