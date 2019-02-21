import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Object tracking.")
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--conf_thres', type=float, default=0.8)
    parser.add_argument('--nms_thres', type=float, default=0.4)
    parser.add_argument('--model',
                        default="Darknet",
                        # default="vgg16-ssd"
                        )
    parser.add_argument("--input_type",
                        default="video",
                        # default="images"
                        )
    args = parser.parse_args()
    return args