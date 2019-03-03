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
from benchmarks.motchallenge.panda_analyze.count_people import get_statistics
import os

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

from object_tracking.track_utils.load_model import load_model_and_classes
from object_tracking.track_utils.parse_args import parse_args
from deeplens_utils.class_mapper import from_mot_id_to_mot_name, mapper


def string_params(params, delimiter=","):
    return delimiter.join([str(param) for param in params])


def run_main(args, index):
    detector, classes, detection_params = load_model_and_classes(args=args)

    # print("classes: ", classes)
    # videopath = '../data/video/overpass.mp4'
    # videopath = os.path.join("videos", "desk.mp4")
    videoname = "desk"
    videofoler = "videos"
    video_ext = ".mp4"
    videopath = videofoler + "/" + videoname + video_ext
    # videoout = videofoler + "/" + videoname + "-track2.mp4"
    videoout = args.output_video + args.bench_case + video_ext

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
        img_folder = "img1"
        # img_folder = "img1_test"  # for test
        images_path = os.path.join(args.images_path, args.bench_case,
                                   img_folder)
        image_files = sorted(os.listdir(images_path))
        last_img_idx = len(image_files)
        img_idx = -1
    else:
        raise Exception(f"Unknown input type: {args.input_type}")

    if args.mot_tracker == "sort_tracker":
        mot_tracker = Sort(min_hits=args.sort_min_hits,
                           max_age=args.sort_max_age)
    elif args.mot_tracker == "iou_tracker":
        mot_tracker = IouTracker()
    else:
        raise Exception(f"Unknown type of the tracker: {args.mot_tracker}")
    tracker_params = mot_tracker.get_params()

    frame_idx = 0
    visibility = 1
    filler = -1
    delimiter = " "

    detection_params_str = string_params(detection_params, "_")
    tracker_params_str = string_params(tracker_params, "_")

    det_ext = ".txt"
    if index == 0:
        out_name = str(
            args.detection_model) + "_" + str(detection_params_str) + "_" + str(
            args.mot_tracker) + "_" + str(tracker_params_str)
        # full_output_path = os.path.join(args.output_path, out_name)
        args.output_path = os.path.join(args.output_path, out_name)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

    full_output_path = os.path.join(args.output_path, args.bench_case + det_ext)

    # The mapper function from the classes recognized by the object detector to
    # the classes recognized by the tracker.
    from_det_class_id_to_track_class_id = mapper.get(
        "from_" + args.from_class + "_id_to_" + args.tracker + "_id")

    with open(full_output_path, "w") as out_csv:
        while (True):
            frame_idx += 1
            # for ii in range(3):t
            if args.input_type == "video":
                ret, frame = vid.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif args.input_type == "image":
                img_idx += 1
                if img_idx >= last_img_idx:
                    break
                image_file = image_files[img_idx]
                image_path = os.path.join(images_path, image_file)
                frame = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            else:
                raise Exception(f"Unknown input type: {args.input_type}")
            detections = detector.detect(frame)

            # Change class ids from the detector one to the tracker.
            class_id_idx = 5  # the index of the class_id value in a detection tuple
            for detection in detections:
                class_det_id = int(detection[class_id_idx])
                class_mot_id = from_det_class_id_to_track_class_id.get(
                    class_det_id)
                detection[class_id_idx] = class_mot_id

            if detections is not None:
                tracked_objects = mot_tracker.track(detections)

                # unique_labels = detections[:, -1].cpu().unique()
                # n_cls_preds = len(unique_labels)
                for obj_id, x1, y1, x2, y2, score, class_id in tracked_objects:
                    box_w = int(x2 - x1)
                    box_h = int(y2 - y1)
                    x1 = int(x1)
                    y1 = int(y1)

                    color = colors[int(obj_id) % len(colors)]
                    color = [i * 255 for i in color]

                    # write the tracks to the output file
                    output = [frame_idx, int(obj_id), x1, y1, box_w, box_h,
                              score, class_id, visibility, filler]

                    class_name = from_mot_id_to_mot_name.get(class_id)

                    output = ",".join(str(x) for x in output) + "\n"
                    # print(output)
                    out_csv.write(output)

                    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h),
                                  color, 4)
                    cv2.rectangle(frame, (x1, y1 - 35),
                                  (x1 + len(class_name) * 19 + 60, y1), color,
                                  -1)
                    cv2.putText(frame, class_name + "-" + str(int(obj_id)),
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

    stats = get_statistics(dataset_path=full_output_path)
    # print(args.bench_case + delimiter + str(
    #     args.detection_model) + delimiter + string_params(
    #     detection_params, delimiter) + delimiter + str(
    #     args.mot_tracker) + delimiter + string_params(
    #     tracker_params,
    #     delimiter) + delimiter + "stats" + delimiter + string_params(
    #     stats, delimiter))

    meta_file = os.path.join(args.output_path, "meta_sql_counters.txt")
    with open(meta_file, "a") as out:
        # print header
        if index == 0:
            out.write(
                "detector: " + args.detection_model + " tracker: " + args.mot_tracker + "\n")
            out.write("bench_case" + delimiter + string_params(
                detection_params[0::2], delimiter) + delimiter + string_params(
                tracker_params[0::2], delimiter) + delimiter + string_params(
                stats[0::2], delimiter) + "\n")

        # print data
        out.write(args.bench_case + delimiter + string_params(
            detection_params[1::2], delimiter) + delimiter + string_params(
            tracker_params[1::2], delimiter) + delimiter + string_params(
            stats[1::2], delimiter) + "\n")

    return stats


if __name__ == "__main__":
    args = parse_args()
    all_stats = []
    for nms_thres in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        args.nms_thres = nms_thres
        for index, bench_case in enumerate(
                ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09",
                 "MOT16-10", "MOT16-11", "MOT16-13"]):
            # for index, bench_case in enumerate(["MOT16-02", "MOT16-04"]):
            args.bench_case = bench_case
            stats = run_main(args, index)
            all_stats.append(stats)
