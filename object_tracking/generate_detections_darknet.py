import os

# import sys
# sys.path.append('../../')
import numpy as np
import cv2
import matplotlib.pyplot as plt

from object_detection.darknet.python.darknet import detect, load_net, load_meta
from object_tracking.utils.parse_args import parse_args

from object_tracking.utils.general_utils import convert_box_xy, convert_box

mot_classes = {"person": 1, "car": 3, "bicycle": 4, "motorbike": 5, "other": 8,
               "track": 3, "bus": 3}
inv_mot_classes = {v: k for k, v in mot_classes.items()}

file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(file_dir, os.path.pardir, "object_detection",
                           "darknet")

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

box_thickness = 4


def run_main(args):
    net = load_net(os.path.join(project_dir, "cfg/yolov3.cfg").encode(),
                   os.path.join(project_dir, "cfg/yolov3.weights").encode(), 0)
    meta = load_meta("config/coco.data".encode())

    with open(args.output_path, "w", newline="") as out:
        frame = 0
        track_id = -1
        distractor_class = 8
        visibility = 1
        print("args.images_path: ", args.images_path)
        image_files = os.listdir(args.images_path)
        for image_file in sorted(image_files):
            print("image_file: ", image_file)
            frame += 1
            path = args.images_path + "/" + image_file
            # image = Image.open(path)
            detections = detect(net, meta, str(path).encode('UTF-8'))
            filler = -1
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            # img_height, img_width, img_channels = image.shape
            for detection in detections:
                # print("detection: ", detection)
                class_pred, class_score, (x1, y1, width, height) = detection
                class_id = mot_classes.get(class_pred.decode(),
                                           distractor_class)
                x1, y1 = convert_box_xy(x1, y1, width, height)
                output = [frame, track_id, x1, y1, width, height, class_score,
                          class_id, visibility, filler]
                output = ",".join(str(x) for x in output) + "\n"
                print(output)
                out.write(output)

                # draw the bounding box for the detection
                color = colors[int(class_id) % len(colors)]
                color = [i * 255 for i in color]

                x1 = int(x1)
                y1 = int(y1)
                width = int(width)
                height = int(height)

                cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height), color,
                              box_thickness)
                class_name = inv_mot_classes.get(int(class_id), "other")
                cv2.rectangle(image, (x1, y1 - 35),
                              (x1 + len(class_name) * 19 + 60, y1),
                              color, -1)
                cv2.putText(image, "class: " + class_name + " track: " + str(
                    int(track_id)),
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # take only the image file name without the extension (e.g., .jpeg)
            image_name = image_file.split(".")[0]
            out_img_name = image_name + "_det.jpg"
            write_path = os.path.join(args.detection_path, out_img_name)
            cv2.imwrite(write_path, image)

            # cv2.imshow(image_file, image)
            # cv2.waitKey()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_main(args)
