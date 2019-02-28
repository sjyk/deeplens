#  DeepLens
#  Copyright (c) 2019. Adam Dziedzic and Sanjay Krishnan
#  Licensed under The MIT License [see LICENSE for details]
#  Written by Adam Dziedzic

import unittest
import logging
import torch
from main.utils import get_logger, set_up_logging
from .iou import IouTracker
import numpy as np


class TestIouTracker(unittest.TestCase):

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

        self.iou = IouTracker(sigma_l=0.3, sigma_h=0.5, sigma_iou=0.3, t_min=5,
                              t_max=2)

    def testIouSuspendedIsItKept1(self):
        """
        Check if a track is kept suspended.
        """
        dets = np.array([[0, 0, 1, 1, 1, 1]])
        expected = [1, 0, 0, 1, 1, 1, 1]

        tracks = self.iou.track(dets)
        np.testing.assert_equal(tracks[0], expected)

        tracks = self.iou.track(dets)
        np.testing.assert_equal(tracks[0], expected)

        tracks = self.iou.track(np.array([]))
        assert len(tracks) == 0

        tracks = self.iou.track(dets)
        np.testing.assert_equal(tracks[0], expected)

if __name__ == '__main__':
    unittest.main()