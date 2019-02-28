import torch
from PIL import Image
import os

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

from object_tracking.track_utils.load_model import load_model_and_classes
from object_tracking.track_utils.parse_args import parse_args


def run_main(args):
    predictor, classes = load_model_and_classes(args=args)

    with open(args.output_path, "w", newline="") as out:
        frame = 0
        track_id = -1
        visibility = 1
        for image_file in os.listdir(args.images_path):
            frame += 1
            image = Image.open(args.images_path + "/" + image_file)
            detections = predictor.predict_for_motchallenge(image)
            for detection in detections:
                x1, y1, x2, y2, object_conf, class_score, class_pred = detection.tolist()
                w = image.size[0]
                h = image.size[1]
                # print("h: ", h, " w: ", w)
                x1 *= (w / args.img_size)
                x2 *= (w / args.img_size)
                y1 *= (h / args.img_size)
                y2 *= (h / args.img_size)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                output = [frame, track_id, x1, x2, width, height,
                          int(class_score), class_pred, visibility]
                output = ",".join(str(x) for x in output) + "\n"
                print(output)
                out.write(output)


if __name__ == "__main__":
    args = parse_args()
    args.scale_image = "yes"
    args.pad_image = "yes"
    run_main(args)
