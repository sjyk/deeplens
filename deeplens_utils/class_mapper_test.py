#  DeepLens
#  Copyright (c) 2019. Adam Dziedzic and Sanjay Krishnan
#  Licensed under The MIT License [see LICENSE for details]
#  Written by Adam Dziedzic

import unittest
import logging
import torch
from main.utils import get_logger, set_up_logging
from .class_mapper import from_coco_id_to_mot_id, from_coco_name_to_coco_id, \
    from_coco_name_to_mot_name, from_mot_name_to_mot_id, \
    from_voc_name_to_voc_id, from_voc_id_to_mot_id
import numpy as np


class ClassMapperTest(unittest.TestCase):

    def setUp(self):
        print("\n")
        log_file = "iou_tracker_test.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")
        seed = 31
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("cuda is available")
            torch.cuda.manual_seed_all(seed)
        else:
            self.device = torch.device("cpu")
            print("cuda is not available")
            torch.manual_seed(seed)
        self.dtype = torch.float
        self.ERR_MESSAGE_ALL_CLOSE = "The expected array desired and " \
                                     "computed actual are not almost equal."

    def test_check_from_one_id_to_another_id_1(self):
        # check coco person
        coco_id = 0
        coco_name = from_coco_name_to_coco_id.get(coco_id)
        mot_name = from_coco_name_to_mot_name.get(coco_name)
        mot_id = from_mot_name_to_mot_id.get(mot_name)
        # mot_id = get_from_coco_id_to_mot_id(coco_id)
        mot_id = from_coco_id_to_mot_id.get(coco_id)
        print("mot_id: ", mot_id)
        assert mot_id == 1

    def test_from_coco_to_mot(self):
        for pair in [("car", "Car"), ("person", "Pedestrian"),
                     ("motorbike", "Motorbike")]:
            coco_name = pair[0]
            mot_name = pair[1]

            coco_id = from_coco_name_to_coco_id.get(coco_name)
            mot_id = from_coco_id_to_mot_id.get(coco_id)
            print("mot_id: ", mot_id)
            mot_expected_id = from_mot_name_to_mot_id.get(mot_name)
            print(f"For coco name {coco_name} with "
                  f"id={coco_id} expected mot name "
                  f"{mot_name} with id {mot_expected_id} and "
                  f"got {mot_id}")
            np.testing.assert_equal(desired=mot_expected_id, actual=mot_id,
                                    err_msg=f"For coco name {coco_name} with "
                                    f"id={coco_id} expected mot name "
                                    f"{mot_name} with id {mot_expected_id} and "
                                    f"got {mot_id}")

    def test_from_voc_to_mot(self):
        for pair in [("car", "Car"), ("person", "Pedestrian"),
                     ("motorbike", "Motorbike"), ("bus", "Car"),
                     ("bicycle", "Bicycle")]:
            voc_name = pair[0]
            mot_name = pair[1]

            voc_id = from_voc_name_to_voc_id.get(voc_name)
            mot_id = from_voc_id_to_mot_id.get(voc_id)
            print("mot_id: ", mot_id)
            mot_expected_id = from_mot_name_to_mot_id.get(mot_name)
            print(f"For voc name {voc_name} with "
                  f"id={voc_id} expected mot name "
                  f"{mot_name} with id {mot_expected_id} and "
                  f"got {mot_id}")
            np.testing.assert_equal(desired=mot_expected_id, actual=mot_id,
                                    err_msg=f"For coco name {voc_name} with "
                                    f"id={voc_id} expected mot name "
                                    f"{mot_name} with id {mot_expected_id} and "
                                    f"got {mot_id}")


if __name__ == '__main__':
    unittest.main()
