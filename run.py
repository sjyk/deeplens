import argparse
import logging
import os
import sys

import cv2

from main.dbms.patchfile import UnclusteredPatchFile
from main.dbms.select import Select
from main.io import FileScan
from main.patch.ssd import SSDPatchGenerator
from main.patch.xform import NullTransformer
from main.utils import get_logger, set_up_logging
from depth_prediction import predictor
from main.dbms.expressions import UDFExpression

DEFAULT_PREDICT_DEPTH_MODEL_PATH = "resources/models/depth_prediction/NYU_" \
                                   "ResNet-UpProj.npy"
DEFAULT_PREDICT_DEPTH_IMAGES_PATH = "resources/demo/image.jpg"

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--is_debug", default=False, type=bool,
                    help="is it the debug mode execution")
parser.add_argument("-l", "--log_file", default="main.log",
                    help="The name of the log file.")
parser.add_argument("-d", "--show_depth_prediction", default=True, type=bool,
                    help="Show the depth prediction for an input image.")
parser.add_argument('-m', '--model_path',
                    default=DEFAULT_PREDICT_DEPTH_MODEL_PATH,
                    help='Converted parameters for the model')
parser.add_argument('-i', '--images_path',
                    default=DEFAULT_PREDICT_DEPTH_IMAGES_PATH,
                    help='Image path or directory of images to predict '
                         'their depth maps')


def run():
    # bulk load images into database
    files_dir = "resources/demo/"
    wait = 0
    if is_debug:
        # process fewer files for the debug mode
        files_dir = "resources/debug/"
        wait = 3000  # milliseconds
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode active")

    f = FileScan(files_dir)
    p = SSDPatchGenerator("resources/models/ssd_mobilenet_v1_coco_2017_11_17/",
                          "object_detection/data/mscoco_label_map.pbtxt", 90)
    h = NullTransformer()
    datastore_folder = "resources/datastore"
    if not os.path.exists(datastore_folder):
        os.makedirs(datastore_folder)
    u = UnclusteredPatchFile("test", f, p, h, datastore_folder + "/data.store")
    u.build()  # can comment out after the datastore is built

    if show_depth_prediction:
        logger.info("Show depth prediction: ")
        predictor.main(model_path=args.model_path, images_path=args.images_path)

    logger.info(
        "Select operator to run basic predicates over the data, visualize all "
        "of the patches that have people in them.")
    s = Select(u, UDFExpression(lambda patch: patch.metadata['tag'] == 'person'))
    for patch in s.read():
        logger.debug("patch metadata: {}".format(patch.metadata['tag']))
        cv2.imshow('image', patch.patch)
        cv2.waitKey(wait)


if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args(sys.argv[1:])
    is_debug = args.is_debug
    log_file = args.log_file
    show_depth_prediction = args.show_depth_prediction

    set_up_logging(log_file=log_file, is_debug=is_debug)
    logger = get_logger(name=__name__)

    run()
