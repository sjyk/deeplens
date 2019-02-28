#  DeepLens
#  Copyright (c) 2019. Adam Dziedzic and Sanjay Krishnan
#  Licensed under The MIT License [see LICENSE for details]
#  Written by Adam Dziedzic

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Object tracking.")

    parser.add_argument('--is_interactive', default="yes",
                        help="Should we display the pictures or only do the "
                             "computation.")

    # Parameters for the object detection with YOLO v3.
    parser.add_argument('--img_size', type=int, default=416,
                        help="The size to which an image is scaled inside the "
                             "YOLO v3 pytorch model.")
    parser.add_argument('--scale_image', default="yes",
                        help="should we scale the images inside YOLO v3 for "
                             "PyTorch, it does not work without this "
                             "pre-processing or at least the changes required "
                             "it to work without this step are too cumbersome "
                             "to be implementeda")
    parser.add_argument('--pad_image', default="yes")
    parser.add_argument('--conf_thres', type=float, default=0.6)
    parser.add_argument('--nms_thres', type=float, default=0.4)

    # Parameters for the object detection with the SSD algorithm that internally
    # uses the VGG network.
    parser.add_argument('--vgg_ssd_top_k', type=int, default=20,
                        help="How many objects detected are returned from SSD "
                             "(starting from the one with the highest "
                             "score/confidence.")
    parser.add_argument('--vgg_ssd_filter_threshold', type=float, default=0.1)
    parser.add_argument('--vgg_ssd_candidate_size', type=int, default=200)
    parser.add_argument('--vgg_ssd_nms_method', type=str, default="soft")

    parser.add_argument('--detection_model',
                        default="Darknet",
                        # default="vgg16-ssd",
                        help="The type of the model for the object detection.")
    parser.add_argument('--mot_tracker',
                        default="sort",
                        # default="iou_tracker",
                        help="The type of the mot (multi object) tracker for "
                             "the object detection.")
    parser.add_argument("--input_type",
                        # default="video",
                        default="image"
                        )
    parser.add_argument("--images_path",
                        default="../benchmarks/motchallenge/MOT16/train/MOT16-02/img1"
                        # default = "../object_detection/darknet_c/data/"
                        )
    parser.add_argument("--detection_path",
                        # default="../benchmarks/motchallenge/MOT16/train/MOT16-02/det_yolo/det.txt"
                        # default="../benchmarks/motchallenge/MOT16/train/MOT16-02/det/det.txt"
                        # default="../object_detection/darknet_c/data/dog_detections.txt"
                        default="../benchmarks/motchallenge/MOT16/train/MOT16-02/det_yolo_sort/det_images/"
                        )
    parser.add_argument("--output_path",
                        # default="../benchmarks/motchallenge/MOT16/train/MOT16-02/det_yolo/det.txt"
                        # default="../benchmarks/motchallenge/MOT16/train/MOT16-02/det_yolo_sort/det.txt"
                        default="../benchmarks/motchallenge/MOT16/train/MOT16-02/det_ssd_sort/"
                        )
    parser.add_argument("--output_video",
                        default="../benchmarks/motchallenge/MOT16/train/MOT16-02/video_det_yolo_sort/yolo_sort.mp4"
                        )

    args = parser.parse_args()
    return args
