import sys
import os

sys.path.append(os.path.abspath('../../'))
from pytorch_ssd.vision.ssd.vgg_ssd import create_vgg_ssd, \
    create_vgg_ssd_predictor

from object_tracking.utils import general_utils
from object_tracking.utils.predictor import Predictor
import torch
from object_tracking.utils.models import Darknet
from object_tracking.utils import general_utils

def load_model_and_classes(args):
    config = "config"
    # Load model and weights.
    if args.model == "Darknet":
        config_path = config + '/yolov3.cfg'
        weights_path = config + '/yolov3.weights'
        class_path = config + '/coco.names'
        classes = general_utils.load_classes(class_path)
        args.num_classes = len(classes)
        model = Darknet(config_path)
        model.load_weights(weights_path)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        predictor = Predictor(args=args, model=model)
    elif args.model == "vgg16-ssd":
        class_path = config + '/voc.names'
        model_path = '../pytorch_ssd/models/vgg16-ssd-mp-0_7726.pth'
        classes = general_utils.load_classes(class_path)
        args.num_classes = len(classes)
        net = create_vgg_ssd(args.num_classes, is_test=True)
        net.load(model_path)
        predictor = create_vgg_ssd_predictor(net, candidate_size=200, top_k=20,
                                             filter_threshold=0.01,
                                             nms_method="soft")
    return predictor, classes