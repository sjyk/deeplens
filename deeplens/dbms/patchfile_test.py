import logging
import os
import unittest

from deeplens.dbms import patchfile
from deeplens.dbms.patchfile import UnclusteredPatchFile
from deeplens.io import FileScan
from deeplens.patch.ssd import SSDPatchGenerator
from deeplens.patch.xform import NullTransformer
from utils import get_logger
from utils import set_up_logging


class TestUnclusteredPatchFile(unittest.TestCase):

    def setUp(self):
        log_file = "../../deeplens.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.debug("Set up test")
        patchfile.logger.setLevel(logging.DEBUG)
        self.resources = "../../resources/"

    def test_build_db(self):
        var = True
        self.assertTrue(var)
        self._build_db()

    def _build_db(self):
        files_dir = self.resources + "debug/"
        scanner = FileScan(files_dir)
        patcher = SSDPatchGenerator(
            self.resources + "models/ssd_mobilenet_v1_coco_2017_11_17/",
            "../../object_detection/data/mscoco_label_map.pbtxt", 90)
        transformer = NullTransformer()

        datastore_folder = self.resources + "datastore"
        if not os.path.exists(datastore_folder):
            os.makedirs(datastore_folder)
        storage_path = datastore_folder + "/data.store"

        db_dir = self.resources + "db_test/"
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        db_file = db_dir + "test"

        u = UnclusteredPatchFile(name=db_file, scanner=scanner, patcher=patcher,
                                 transformer=transformer,
                                 storage_path=storage_path)
        u.build()
        return u

    def test_read_db(self):
        self.logger.debug("test read db")
        db = self._build_db()
        for patch in db.read():
            self.logger.debug(
                "patch metadata: {}".format(patch.metadata['tag']))


if __name__ == '__main__':
    unittest.main()
