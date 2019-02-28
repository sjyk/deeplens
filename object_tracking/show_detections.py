import os
import csv
from object_tracking.track_utils.parse_args import parse_args
from object_tracking.track_utils.general_utils import convert_box, \
    convert_box_to_cv2_rectangle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mot_classes = {"person": 1, "person_on_vehicle": 2, "car": 3, "bicycle": 4,
               "motorbike": 5, "other": 8, "pedestrian": -1}
inv_mot_classes = {v: k for k, v in mot_classes.items()}

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

img_name_size = 6
box_thickness = 4

videoname = "MOT16-02"
videofoler = "videos"
videoout = videofoler + "/" + videoname + "-track-ground-truth-only-people.mp4"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(videoout, fourcc, 29.0, (1920, 1080))


def run_main(args):
    print(os.getcwd())
    detection_path = args.detection_path
    detection_name = detection_path.split('/')[-1]
    frame = 1
    # data frame
    df = pd.read_csv(args.detection_path, sep=",", header=None,
                             names=['frame', 'id', 'x', 'y', 'w', 'h',
                                    'score', 'class', 'wy', 'wz'])
    df = df.sort_values(by=['frame'])
    # extract only people
    df = df[(df['class'] == 1) | (df['class'] == 2) | (df['class'] == 7)]
    print(df.head(100))
    img_name = str(frame).rjust(img_name_size, '0')
    img_path = args.images_path + "/" + img_name + ".jpg"
    # image = Image.open(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_height, img_width, img_channels = image.shape
    for row in df.itertuples(index=False):
        frame_read, track_id, x1, y1, width, height, class_score, class_id, visibility, _ = row
        x1 = int(float(x1))
        y1 = int(float(y1))
        width = int(float(width))
        height = int(float(height))
        if int(frame_read) > frame:
            if args.is_interactive == "yes":
                cv2.imshow(detection_name, image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                cv2.waitKey()
            out_img_name = img_name + "_detections.jpg"
            write_path = os.path.join(args.detection_path, os.pardir,
                                      out_img_name)
            cv2.imwrite(write_path, image)
            out.write(image)
            frame += 1
            img_name = str(frame).rjust(img_name_size, '0')
            img_path = args.images_path + "/" + img_name + ".jpg"
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        color = colors[int(class_id) % len(colors)]
        color = [i * 255 for i in color]

        # left, top, right, bottom = convert_box(x1, y1, width, height,
        #                                        img_width, img_height)
        left, top, right, bottom = convert_box_to_cv2_rectangle(
            x1, y1, width, height, img_width, img_height)

        # print("left, top, right, bottom: ", left, top, right, bottom)
        cv2.rectangle(image, (left, top), (right, bottom), color,
                      box_thickness)
        class_name = inv_mot_classes.get(int(class_id), "other")
        cv2.rectangle(image, (x1, y1 - 35),
                      (x1 + len(class_name) * 19 + 60, y1),
                      color, -1)
        cv2.putText(image, "track_id: " + str(
            int(track_id)) + "class: " + class_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    out_img_name = img_name + "_detections.jpg"
    write_path = os.path.join(args.detection_path, os.pardir, out_img_name)
    cv2.imwrite(write_path, image)
    out.write(image)

    if args.is_interactive == "yes":
        cv2.imshow(detection_name, image)
        cv2.waitKey()
    cv2.destroyAllWindows()
    out.release()

if __name__ == "__main__":
    args = parse_args()
    args.is_interactive="no"
    # args.detection_path = "../../motchallenge/MOT16/train/MOT16-02/det_yolo/det.txt"
    # args.detection_path = "../../motchallenge/res/MOT16/iou_tracker/MOT16-02_yolo.txt"
    args.detection_path = "../../motchallenge/MOT16/train/MOT16-02/gt/gt.txt"
    run_main(args)
