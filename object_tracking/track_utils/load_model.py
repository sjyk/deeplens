import sys
import os

sys.path.append(os.path.abspath('../../'))
from object_detection.pytorch_ssd.vision.ssd.vgg_ssd import create_vgg_ssd, \
    create_vgg_ssd_predictor

from object_detection.darknet_pytorch.detector_darknet_pytorch import \
    DetectorDarknetPytorch
import torch
from object_detection.darknet_pytorch.models import Darknet
from object_tracking.track_utils import general_utils


def load_model_and_classes(args):
    config = "config"
    # Load model and weights.
    if args.detection_model == "Darknet":
        config_path = config + '/yolov3.cfg'
        weights_path = config + '/yolov3.weights'
        class_path = config + '/coco.names'
        classes = general_utils.load_classes(class_path)
        args.num_classes = len(classes)
        args.from_class = "coco"
        model = Darknet(config_path)
        model.load_weights(weights_path)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        detector = DetectorDarknetPytorch(args=args, model=model)
        params = ["nms_thres", str(args.nms_thres), "conf_thres",
                  str(args.conf_thres)]
    elif args.detection_model == "vgg16-ssd":
        class_path = config + '/voc.names'
        model_path = '../object_detection/pytorch_ssd/models/vgg16-ssd-mp-0_7726.pth'
        classes = general_utils.load_classes(class_path)
        args.num_classes = len(classes)
        args.from_class = "voc"
        net = create_vgg_ssd(args.num_classes, is_test=True)
        net.load(model_path)
        detector = create_vgg_ssd_predictor(
            net, candidate_size=args.vgg_ssd_candidate_size,
            top_k=args.vgg_ssd_top_k,
            filter_threshold=args.vgg_ssd_filter_threshold,
            nms_method=args.vgg_ssd_nms_method)
        params = ["top_k", str(
            args.vgg_ssd_top_k), "filter_threshold", str(
            args.vgg_ssd_filter_threshold), "candidate_size", str(
            args.vgg_ssd_candidate_size), "nms_method", str(
            args.vgg_ssd_nms_method)]
    return detector, classes, params
