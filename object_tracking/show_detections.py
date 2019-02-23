import os
import csv
from object_tracking.utils.parse_args import parse_args
import cv2
import matplotlib.pyplot as plt
import numpy as np

mot_classes = {"person": 1, "car": 3, "bicycle": 4, "motorbike": 5, "other": 8,
               "pedestrian": -1}
inv_mot_classes = {v: k for k, v in mot_classes.items()}

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

img_name_size = 6
box_thickness = 4


def convert_box(x1, y1, width, height, img_width, img_height):
    """
    Convert from x1, y1, representing the center of the box to the top left and
    bottom right coordinates.

    :param x1:
    :param y1:
    :param width:
    :param height:
    :param img_width:
    :param img_height:
    :return:
    """
    left = (x1 - width // 2)
    right = (x1 + width // 2)
    top = (y1 - height // 2)
    bot = (y1 + height // 2)

    if left < 0: left = 0
    if right > img_width - 1: right = img_width - 1
    if top < 0: top = 0;
    if bot > img_height - 1: bot = img_height - 1

    return left, top, right, bot


def run_main(args):
    print(os.getcwd())
    detection_path = args.detection_path
    detection_name = detection_path.split('/')[-1]
    with open(args.detection_path, "r") as csv_file:
        frame = 1
        csv_reader = csv.reader(csv_file, delimiter=',')
        img_name = str(frame).rjust(img_name_size, '0')
        img_path = args.images_path + "/" + img_name + ".jpg"
        # image = Image.open(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_height, img_width, img_channels = image.shape
        for row in csv_reader:
            frame_read, track_id, x1, y1, width, height, class_score, class_id, visibility, _ = row
            x1 = int(float(x1))
            y1 = int(float(y1))
            width = int(float(width))
            height = int(float(height))
            if int(frame_read) > frame:
                cv2.imshow(detection_name, image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                cv2.waitKey()
                out_img_name = img_name + "_detections.jpg"
                write_path = os.path.join(args.detection_path, os.pardir,
                                          out_img_name)
                cv2.imwrite(write_path, image)
                frame += 1
                img_name = str(frame).rjust(img_name_size, '0')
                img_path = args.images_path + "/" + img_name + ".jpg"
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)

            color = colors[int(class_id) % len(colors)]
            color = [i * 255 for i in color]

            left, top, right, bottom = convert_box(x1, y1, width, height,
                                                   img_width, img_height)

            # print("left, top, right, bottom: ", left, top, right, bottom)
            cv2.rectangle(image, (left, top), (right, bottom), color,
                          box_thickness)
            class_name = inv_mot_classes.get(int(class_id), "other")
            cv2.rectangle(image, (x1, y1 - 35),
                          (x1 + len(class_name) * 19 + 60, y1),
                          color, -1)
            cv2.putText(image, "class: " + class_name + " track: " + str(
                int(track_id)),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.imshow(detection_name, image)
        cv2.waitKey()
        out_img_name = img_name + "_detections.jpg"
        write_path = os.path.join(args.detection_path, os.pardir, out_img_name)
        cv2.imwrite(write_path, image)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_main(args)
