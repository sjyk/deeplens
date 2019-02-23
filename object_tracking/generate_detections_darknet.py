import os

# import sys
# sys.path.append('../../')

from object_detection.darknet.python.darknet import detect, load_net, load_meta
from object_tracking.utils.parse_args import parse_args

mot_classes = {"person": 1, "car": 3, "bicycle": 4, "motorbike": 5, "other": 8,
               "track": 3, "bus": 3}

file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(file_dir, os.path.pardir, "object_detection",
                           "darknet")


def run_main(args):
    net = load_net(os.path.join(project_dir, "cfg/yolov3.cfg").encode(),
                   os.path.join(project_dir, "cfg/yolov3.weights").encode(), 0)
    meta = load_meta("config/coco.data".encode())

    with open(args.output_path, "w", newline="") as out:
        frame = 0
        track_id = -1
        distractor_class = 8
        visibility = 1
        for image_file in os.listdir(args.images_path):
            frame += 1
            path = args.images_path + "/" + image_file
            # image = Image.open(path)
            detections = detect(net, meta, str(path).encode('UTF-8'))
            filler = -1
            for detection in detections:
                # print("detection: ", detection)
                class_pred, class_score, (x1, y1, width, height) = detection
                class_id = mot_classes.get(class_pred.decode(),
                                           distractor_class)
                output = [frame, track_id, x1, y1, width, height, class_score,
                          class_id, visibility, filler]
                output = ",".join(str(x) for x in output) + "\n"
                print(output)
                out.write(output)


if __name__ == "__main__":
    args = parse_args()
    run_main(args)
