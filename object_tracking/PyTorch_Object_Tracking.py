#  MIT License
#
#  Copyright (c) 2019. Adam Dziedzic and Sanjay Krishnan
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
#  Written by Adam Dziedzic

import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from object_tracking.sort_tracker.sort import Sort
from object_tracking.iou_tracker.iou import IouTracker
import os

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

from object_tracking.utils.load_model import load_model_and_classes
from object_tracking.utils.parse_args import parse_args
from object_tracking.utils.mot_utils import get_mot_class_id


def run_main(args):
    detector, classes, detection_params = load_model_and_classes(args=args)

    # videopath = '../data/video/overpass.mp4'
    # videopath = os.path.join("videos", "desk.mp4")
    videoname = "desk"
    videofoler = "videos"
    videopath = videofoler + "/" + videoname + ".mp4"
    # videoout = videofoler + "/" + videoname + "-track2.mp4"
    videoout = args.output_video

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'MP42')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(videoout, fourcc, 29.0, (1920, 1080))

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # initialize Sort object and video capture
    if args.input_type == "video":
        vid = cv2.VideoCapture(videopath)
        print("width, height, FPS, FFCV: ", vid.get(cv2.CAP_PROP_FRAME_WIDTH),
              vid.get(cv2.CAP_PROP_FRAME_HEIGHT), vid.get(cv2.CAP_PROP_FPS),
              vid.get(cv2.CAP_PROP_FOURCC))
    elif args.input_type == "image":
        image_files = sorted(os.listdir(args.images_path))
        last_img_idx = len(image_files)
        img_idx = 0
    else:
        raise Exception(f"Unknown input type: {args.input_type}")

    if args.mot_tracker == "sort":
        mot_tracker = Sort()
    elif args.mot_tracker == "iou_tracker":
        mot_tracker = IouTracker()
    else:
        raise Exception(f"Unknown type of the tracker: {args.mot_tracker}")

    frame_idx = 0
    visibility = 1
    filler = -1
    out_name = "det_" + str(args.detection_model) + detection_params + ".txt"
    full_output_path = os.path.join(args.output_path, out_name)

    with open(full_output_path, "w") as out_csv:
        while (True):
            frame_idx += 1
            # for ii in range(3):t
            if args.input_type == "video":
                ret, frame = vid.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif args.input_type == "image":
                if img_idx >= last_img_idx:
                    break
                image_file = image_files[img_idx]
                image_path = os.path.join(args.images_path, image_file)
                frame = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                img_idx += 1
            else:
                raise Exception(f"Unknown input type: {args.input_type}")
            detections = detector.detect(frame)

            if detections is not None:
                tracked_objects = mot_tracker.update(detections)

                # unique_labels = detections[:, -1].cpu().unique()
                # n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, obj_id, class_id, score in tracked_objects:
                    box_w = int(x2 - x1)
                    box_h = int(y2 - y1)
                    x1 = int(x1)
                    y1 = int(y1)

                    color = colors[int(obj_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cls = classes[int(class_id)]

                    # write the tracks to the output file
                    output = [frame_idx, int(obj_id), x1, y1, box_w, box_h,
                              score, get_mot_class_id(cls), visibility, filler]
                    output = ",".join(str(x) for x in output) + "\n"
                    print(output)
                    out_csv.write(output)

                    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h),
                                  color, 4)
                    cv2.rectangle(frame, (x1, y1 - 35),
                                  (x1 + len(cls) * 19 + 60, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(int(obj_id)),
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            if args.is_interactive == "yes":
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            out.write(frame)

        # plt.figure(figsize=(12, 8))
        # plt.title("Video Stream")
        # plt.imshow(frame)
        # plt.show()
        # plt.close()
        # clear_output(wait=False)
    out.release()
    if args.is_interactive == "yes":
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_main(args)
