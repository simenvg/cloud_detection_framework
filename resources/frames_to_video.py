import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Input path to darknet')
parser.add_argument('data_path', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
parser.add_argument('output_path', type=str, nargs=1,
                    help='Set path to data folder, containg datasets')
args = parser.parse_args()
output_path = args.output_path[0]
data_path = args.data_path[0]


images = os.listdir(data_path)
images.sort(key=lambda f: int(filter(str.isdigit, f)))


# Determine the width and height from the first image
image_path = os.path.join(data_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video', frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

for image in images:

    image_path = os.path.join(data_path, image)
    frame = cv2.imread(image_path)

    out.write(frame)  # Write out frame to video

    cv2.imshow('video', frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output_path))


print(images)
