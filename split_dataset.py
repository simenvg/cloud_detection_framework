import os
import sys
import random


def change_folder(current_path, new_path, filename):
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    os.rename(os.path.join(current_path, filename),
              os.path.join(new_path, filename))


def split_to_train_and_test_folders(path):
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            if random.random() >= 0.50:
                change_folder(path, os.path.join(path, 'train'), filename)
                change_folder(path, os.path.join(path, 'train'), filename[:-4] + '.xml')
            else:
                change_folder(path, os.path.join(path, 'test'), filename)
                change_folder(path, os.path.join(path, 'test'), filename[:-4] + '.xml')
        else:
            continue


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Argument = Complete path to image folder")
        sys.exit()
    path = sys.argv[1]
    split_to_train_and_test_folders(path)
