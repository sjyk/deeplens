from __future__ import division
import math
import torch
import numpy as np


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def convert_box_xy(x1, y1, width, height):
    """
    Convert from x1, y1, representing the center of the box to the top left
    coordinate (corner).

    :param x1: the x coordinate for the center of the bounding box
    :param y1: the y coordinate for the center of the bounding box
    :param width: with of the bounding box
    :param height: height of the bounding box
    :param img_width: the width of the image
    :param img_height: the height of the image
    :return: the top left coordinate (corner) of the bounding box
    """
    left = (x1 - width // 2)
    top = (y1 - height // 2)

    if left < 0: left = 0
    if top < 0: top = 0;

    return left, top


def convert_box(x1, y1, width, height, img_width, img_height):
    """
    Convert from x1, y1, representing the center of the box and its width and
    height to the top left and bottom right coordinates.

    :param x1: the x coordinate for the center of the bounding box
    :param y1: the y coordinate for the center of the bounding box
    :param width: with of the bounding box
    :param height: height of the bounding box
    :param img_width: the width of the image
    :param img_height: the height of the image
    :return: the top left and bottomg right coordinates (corner) of the
    bounding box.
    """
    left = (x1 - width // 2)
    right = (x1 + width // 2)
    top = (y1 - height // 2)
    bot = (y1 + height // 2)

    if left < 0: left = 0
    if right > img_width - 1: right = img_width - 1
    if top < 0: top = 0;
    if bot > img_height - 1: bot = img_height - 1

    return left, top, right, bot


def convert_box_to_cv2_rectangle(x1, y1, width, height, img_width, img_height):
    """
    Change the x1, y1 top left coordinates of the bounding box and its width
    and height to the top left and bottom right corners of the bounding box for
    the cv2 rectangle function.

    :param x1: the x coordinate for the top left corner of the bounding box
    :param y1: the y coordinate for the bottom right corner of the bounding box
    :param width: with of the bounding box
    :param height: height of the bounding box
    :param img_width: the width of the image
    :param img_height: the height of the image
    :return: top left and bottom right corners for cv2 rectangle.
    """
    left = int(x1)
    top = int(y1)
    right = left + int(width)
    if right >= img_width:
        right = img_width - 1
    bottom = y1 + int(height)
    if height >= img_height:
        bottom = img_height - 1
    return left, top, right, bottom


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        return fp.read().split("\n")


def load_classes_strip(path):
    """
    Loads class labels at 'path' and strip them.
    """
    return [name.strip() for name in open(path).readlines()]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap