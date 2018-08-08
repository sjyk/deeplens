from deeplens.io import FileScan
from deeplens.patch.core import NullPatchGenerator
from deeplens.patch.ssd import SSDPatchGenerator
from deeplens.patch.xform import NullTransformer
from deeplens.dbms.patchfile import UnclusteredPatchFile
from deeplens.dbms.select import Select

import cv2

#bulk load images into database
f = FileScan("resources/demo/")
p = SSDPatchGenerator("resources/models/ssd_mobilenet_v1_coco_2017_11_17/", "object_detection/data/mscoco_label_map.pbtxt", 90)
h = NullTransformer()
u = UnclusteredPatchFile("test", f,p,h, "resources/datastore/data.store")
u.build() #can comment out after the datastore is built

#select operator to run basic predicates over the data
s = Select(u, lambda patch: patch.metadata['tag'] == 'person')
for patch in s.read():
    cv2.imshow('image',patch.patch)
    cv2.waitKey(0)

