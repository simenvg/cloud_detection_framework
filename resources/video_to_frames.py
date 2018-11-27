import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('video_path', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
parser.add_argument('output_path', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
args = parser.parse_args()
output_path = args.output_path[0]
video_path = args.video_path[0]


vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(os.path.join(output_path, "frame%d.jpg" %
                             count), image)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
