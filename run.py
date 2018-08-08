import argparse
import logging
import os
import sys

import cv2

from deeplens.dbms.patchfile import UnclusteredPatchFile
from deeplens.dbms.select import Select
from deeplens.io import FileScan
from deeplens.patch.ssd import SSDPatchGenerator
from deeplens.patch.xform import NullTransformer
from utils import get_logger, set_up_logging

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--is_debug", default=False, type=bool,
                    help="is it the debug mode execution")
parser.add_argument("-l", "--log_file", default="deeplens.log",
                    help="The name of the log file.")


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

    # select operator to run basic predicates over the data, visualize all of
    # the patches that have people in them
    s = Select(u, lambda patch: patch.metadata['tag'] == 'person')
    for patch in s.read():
        logger.debug("patch metadata: {}".format(patch.metadata['tag']))
        cv2.imshow('image', patch.patch)
        cv2.waitKey(wait)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    is_debug = args.is_debug
    log_file = args.log_file
    set_up_logging(log_file=log_file)
    logger = get_logger(name=__name__)

    run()
