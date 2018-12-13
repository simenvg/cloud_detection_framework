import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('results_path', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
args = parser.parse_args()
results_path = args.results_path[0]


def load_file(file_path):
    file = open(file_path, 'r')
    prec_recall_dict = {}
    prec_recall_dict['name'] = file.readline()
    prec_recall_dict['precisions'] = file.readline().split(' ')
    prec_recall_dict['recalls'] = file.readline().split(' ')
    file.close()
    return prec_recall_dict


def get_files(results_path):
    prec_recall_paths = []
    trained_on = ['trained_on_trf_bc_bf',
                  'trained_on_trf_bc_bf_bbnb', 'trained_on_trf_bc_bf_bb_build']
    tested_on = ['tested_on_b_and_b', 'tested_on_trf',
                 'tested_on_bc_bf', 'tested_on_bc_bf_trf']
    ssd_yolo = ['results-yolo', 'results-ssd']

    while True:
        print('Trained on dataset:')
        for i in range(len(trained_on)):
            print('[' + str(i) + ']  : ' + trained_on[i])
        print('Tested on dataset:')
        for i in range(len(tested_on)):
            print('[' + str(i) + ']  : ' + tested_on[i])
        print('Detection alg:')
        for i in range(len(ssd_yolo)):
            print('[' + str(i) + ']  : ' + ssd_yolo[i])
        user_input = raw_input(
            'Type number of trained on space, then number tested on, then ssd or yolo. To quit press q.  ')
        if user_input == 'q':
            break
        user_input_list = user_input.split(' ')
        user_input_name = raw_input('name in plot: ')

        prec_recall_paths.append((os.path.join(
            results_path, trained_on[int(user_input_list[0])], tested_on[int(user_input_list[1])], ssd_yolo[int(user_input_list[2])], 'prec_recalls.txt'), user_input_name))
    return (prec_recall_paths)


# def get_files2(results_path):
#     i = 1
#     trained_on = ['trained_on_trf_bc_bf',
#                   'trained_on_trf_bc_bf_bb_build', 'trained_on_trf_bc_bf_bbnb']
#     tested_on = ['tested_on_b_and_b', 'tested_on_trf',
#                  'tested_on_bc_bf', 'tested_on_bc_bf_trf']
#     ssd_yolo = ['results-yolo', 'results-ssd']
#     for train in trained_on:
#         for test in tested_on:
#             print(train, test)
#             prec_recall_out_paths = [os.path.join(results_path, train, test, ssd_yolo[0], 'prec_recalls.txt'), os.path.join(
#                 results_path, train, test, ssd_yolo[1], 'prec_recalls.txt')]
#             plot_prec_recall(prec_recall_out_paths, os.path.join(
#                 results_path, train, test, 'prec_recall.png'), i)
#             i += 1


def plot_prec_recall(prec_recall_file_paths, output_path, i):
    print('paths: ', prec_recall_file_paths)
    for file_path in prec_recall_file_paths:
        prec_recall_dict = load_file(file_path[0])
        name = file_path[1]
        precisions = prec_recall_dict['precisions']
        recalls = prec_recall_dict['recalls']
        plt.figure(i)
        plt.plot(recalls, precisions, label=name, linewidth=3)
        plt.grid(True)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(top=1.03)
        plt.ylim(bottom=0)
        plt.xlim(right=1)
        plt.xlim(left=0)
        plt.rcParams.update({'font.size': 20})
        plt.rcParams["figure.dpi"] = 1000
        plt.legend(loc="lower left")
        # leg_lines = leg.get_lines()
        # plt.setp(leg_lines, linewidth=3)
        plt.tight_layout()
        plt.title('Precision/Recall curve')

    plt.show()
    # plt.savefig(output_path)


output_path = '/home/simenvg/'

files = get_files(results_path)
plot_prec_recall(files, output_path, 1)
