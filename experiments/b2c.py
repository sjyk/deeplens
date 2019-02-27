import argparse
import logging
import os
import sys
import inspect
import cv2
import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from main.dbms.patchfile import UnclusteredPatchFile, BTreeIndex
from main.dbms.select import Select
from main.io import VideoScan
from main.patch.ssd import SSDPatchGenerator
from main.patch.xform import NullTransformer
from main.dbms.expressions import EqualityExpression

RESULTS_FILE = 'results.csv'
files_dir = "resources/b2/b2.mp4"

def buildStore():

    if os.path.exists("b2c"): 
        os.remove("b2c") 

    start_time = datetime.datetime.now()

    f = VideoScan(files_dir, resize=0.2, sampling=1000)
    p = SSDPatchGenerator("resources/models/ssd_mobilenet_v1_coco_2017_11_17/",
                          "object_detection/data/mscoco_label_map.pbtxt", 90,confidence=0.10)
    h = NullTransformer()
    u = UnclusteredPatchFile("b2c", f, p, h, "resources/data.store")
    u.build()

    i = BTreeIndex(u, "b2c_hash", "tag")
    i.build()

    end_time = (datetime.datetime.now() - start_time).total_seconds()
    size = os.stat('b2c').st_size

    f = open(RESULTS_FILE,'a')
    f.write("b2c,build,"+str(end_time)+","+str(size)+",24:30\n")
    f.close()

    return i


def processQuery(src):
    start_time = datetime.datetime.now()

    s = Select(src, EqualityExpression('tag','person'))

    [p for p in s.read()]

    end_time = (datetime.datetime.now() - start_time).total_seconds()

    f = open(RESULTS_FILE,'a')
    f.write("b2c,query,"+str(end_time)+","+str(0)+",24:30\n")
    f.close()


def run():
    u = buildStore()
    processQuery(u)


if __name__ == "__main__":
    # Parse arguments
    run()
