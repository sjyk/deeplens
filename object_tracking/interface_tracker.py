#  DeepLens
#  Copyright (c) 2019. Adam Dziedzic and Sanjay Krishnan
#  Licensed under The MIT License [see LICENSE for details]
#  Written by Adam Dziedzic

import abc

class Tracker(abc.ABC):

    @abc.abstractmethod
    def track(self, dets):
        """
        Track object in frames.
        :param dets: detections provided by an object detector. The format of
        detections is specified in the detector interface.

        :return: Returned tracks are in the form:
        [[obj_id, x1, y1, x2, y2, score, class_id],
        [obj_id, x1, y1, x2, y2, score, class_id],...]

        - score is also called conf_level or confidence score
        - class_id is also called label or class_pred
        """
        pass
