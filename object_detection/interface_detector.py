import abc

class Detector(abc.ABC):

    @abc.abstractmethod
    def detect(self, img):
        """
        Detect object in the image.
        :return: detections, each in the following format:
        (x1, y1, box_w, box_h, conf_level, class_pred)
        """
        pass