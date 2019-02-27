import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Object tracking.")
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--scale_image', default="yes")
    parser.add_argument('--pad_image', default="yes")
    parser.add_argument('--is_interactive', default="yes")
    parser.add_argument('--conf_thres', type=float, default=0.6)
    parser.add_argument('--nms_thres', type=float, default=0.4)
    parser.add_argument('--model',
                        default="Darknet",
                        # default="vgg16-ssd"
                        )
    parser.add_argument("--input_type",
                        # default="video",
                        default="image"
                        )
    parser.add_argument("--images_path",
                        default="../../motchallenge/MOT16/train/MOT16-02/img1"
                        # default = "../object_detection/darknet/data/"
                        )
    parser.add_argument("--detection_path",
                        # default="../../motchallenge/MOT16/train/MOT16-02/det_yolo/det.txt"
                        # default="../../motchallenge/MOT16/train/MOT16-02/det/det.txt"
                        # default="../object_detection/darknet/data/dog_detections.txt"
                        default="../../motchallenge/MOT16/train/MOT16-02/det_yolo_sort/det_images/"
                        )
    parser.add_argument("--output_path",
                        # default="../../motchallenge/MOT16/train/MOT16-02/det_yolo/det.txt"
                        default = "../../motchallenge/MOT16/train/MOT16-02/det_yolo_sort/det.txt"
                        )
    parser.add_argument("--output_video",
                        default="../../motchallenge/MOT16/train/MOT16-02/video_det_yolo_sort/yolo_sort.mp4"
                        )

    args = parser.parse_args()
    return args
