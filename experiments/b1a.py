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

from deeplens.dbms.patchfile import UnclusteredPatchFile
from deeplens.dbms.join import NestedLoopJoin
from deeplens.io import FileScan
from deeplens.patch.core import NullPatchGenerator
from deeplens.patch.xform import ColorHistTransformer
from deeplens.dbms.expressions import ImageMatchExpression



RESULTS_FILE = 'results.csv'
files_dir = "resources/b1/"

def buildStore():

    if os.path.exists("b1a"): 
        os.remove("b1a") 

    start_time = datetime.datetime.now()

    f = FileScan(files_dir)
    p = NullPatchGenerator()
    h = ColorHistTransformer()
    u = UnclusteredPatchFile("b1a", f, p, h, "resources/data.store")
    u.build()

    end_time = (datetime.datetime.now() - start_time).total_seconds()
    size = os.stat('b1a').st_size

    f = open(RESULTS_FILE,'a')
    f.write("b1a,build,"+str(end_time)+","+str(size)+",779\n")
    f.close()

    return u


def processQuery(src):
    start_time = datetime.datetime.now()

    nl = NestedLoopJoin(src, src, ImageMatchExpression(1e-6))

    [p for p in nl.read()]

    end_time = (datetime.datetime.now() - start_time).total_seconds()

    f = open(RESULTS_FILE,'a')
    f.write("b1a,query,"+str(end_time)+","+str(0)+",779\n")
    f.close()


def run():
    u = buildStore()
    processQuery(u)


if __name__ == "__main__":
    # Parse arguments
    run()
