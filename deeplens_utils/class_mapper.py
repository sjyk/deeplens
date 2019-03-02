#  DeepLens
#  Copyright (c) 2019. Adam Dziedzic and Sanjay Krishnan
#  Licensed under The MIT License [see LICENSE for details]
#  Written by Adam Dziedzic

"""
Map class names and class id-s between different benchmarks.
"""


def invert_dict(dict):
    return {v: k for k, v in dict.items()}


# MOT
mot_other_id = 8
mot_other_name = "other"
from_mot_name_to_mot_id = {"Pedestrian": 1,
                           "Person on vehicle": 2,
                           "Car": 3,
                           "Bicycle": 4,
                           "Motorbike": 5,
                           "Non motorized vehicle": 6,
                           "Static person": 7,
                           "Distractor": 8,
                           "Occluder": 9,
                           "Occluder on the ground": 10,
                           "Occluder full": 11,
                           "Reflection": 12}
from_mot_id_to_mot_name = invert_dict(from_mot_name_to_mot_id)


def get_from_mot_name_to_mot_id(name):
    return from_mot_name_to_mot_id.get(name, mot_other_id)


def get_from_mot_id_to_mot_name(id):
    return from_mot_id_to_mot_name.get(id, mot_other_name)


# COCO
coco_classes_array = ["person", "bicycle", "car", "motorbike", "aeroplane",
                      "bus", "train", "truck", "boat", "traffic light",
                      "fire hydrant", "stop sign", "parking meter", "bench",
                      "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                      "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis",
                      "snowboard", "sports ball", "kite", "baseball bat",
                      "baseball glove", "skateboard", "surfboard",
                      "tennis racket", "bottle", "wine glass", "cup", "fork",
                      "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                      "orange", "broccoli", "carrot", "hot dog", "pizza",
                      "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                      "remote", "keyboard", "cell phone", "microwave", "oven",
                      "toaster", "sink", "refrigerator", "book", "clock",
                      "vase", "scissors", "teddy bear", "hair drier",
                      "toothbrush"]
from_coco_id_to_coco_name = {k: v for k, v in enumerate(coco_classes_array)}
from_coco_name_to_coco_id = invert_dict(from_coco_id_to_coco_name)

# VOC
voc_classes_array = ["BACKGROUND", "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair", "cow",
                     "diningtable", "dog", "horse", "motorbike", "person",
                     "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
from_voc_id_to_voc_name = {k: v for k, v in enumerate(voc_classes_array)}
from_voc_name_to_voc_id = invert_dict(from_voc_id_to_voc_name)

default_mot = "Distractor"

# From COCO to MOT
from_coco_name_to_mot_name = {"person": "Pedestrian",
                              "bicycle": "Bicycle",
                              "car": "Car",
                              "motorbike": "Motorbike",
                              "aeroplane": default_mot,
                              "bus": "Car",
                              "train": default_mot,
                              "truck": "Car",
                              "boat": default_mot,
                              "traffic light": default_mot,
                              "fire hydrant": default_mot,
                              "stop sign": default_mot,
                              "parking meter": default_mot,
                              "bench": default_mot,
                              "bird": default_mot,
                              "cat": default_mot,
                              "dog": default_mot,
                              "horse": default_mot,
                              "sheep": default_mot,
                              "cow": default_mot,
                              "elephant": default_mot,
                              "bear": default_mot,
                              "zebra": default_mot,
                              "giraffe": default_mot,
                              "backpack": default_mot,
                              "umbrella": default_mot,
                              "handbag": default_mot,
                              "tie": default_mot,
                              "suitcase": default_mot,
                              "frisbee": default_mot,
                              "skis": default_mot,
                              "snowboard": default_mot,
                              "sports ball": default_mot,
                              "kite": default_mot,
                              "baseball bat": default_mot,
                              "baseball glove": default_mot,
                              "skateboard": default_mot,
                              "surfboard": default_mot,
                              "tennis racket": default_mot,
                              "bottle": default_mot,
                              "wine glass": default_mot,
                              "cup": default_mot,
                              "fork": default_mot,
                              "knife": default_mot,
                              "spoon": default_mot,
                              "bowl": default_mot,
                              "banana": default_mot,
                              "apple": default_mot,
                              "sandwich": default_mot,
                              "orange": default_mot,
                              "broccoli": default_mot,
                              "carrot": default_mot,
                              "hot dog": default_mot,
                              "pizza": default_mot,
                              "donut": default_mot,
                              "cake": default_mot,
                              "chair": default_mot,
                              "sofa": default_mot,
                              "pottedplant": default_mot,
                              "bed": default_mot,
                              "diningtable": default_mot,
                              "toilet": default_mot,
                              "tvmonitor": default_mot,
                              "laptop": default_mot,
                              "mouse": default_mot,
                              "remote": default_mot,
                              "keyboard": default_mot,
                              "cell phone": default_mot,
                              "microwave": default_mot,
                              "oven": default_mot,
                              "toaster": default_mot,
                              "sink": default_mot,
                              "refrigerator": default_mot,
                              "book": default_mot,
                              "clock": default_mot,
                              "vase": default_mot,
                              "scissors": default_mot,
                              "teddy bear": default_mot,
                              "hair drier": default_mot,
                              "toothbrush": default_mot
                              }
from_mot_name_to_coco_name = invert_dict(from_coco_name_to_mot_name)


def get_from_coco_id_to_mot_id(coco_id):
    # coco_name = from_coco_id_to_coco_name.get(coco_id)
    # mot_name = from_coco_name_to_mot_name.get(coco_name)
    # mot_id = from_mot_name_to_mot_id.get(mot_name)
    return from_mot_name_to_mot_id.get(
        from_coco_name_to_mot_name.get(from_coco_id_to_coco_name.get(coco_id)))


def get_dict_from_coco_id_to_mot_id():
    dict = {}
    for coco_id in range(len(coco_classes_array)):
        dict[coco_id] = get_from_coco_id_to_mot_id(coco_id)
    return dict


def get_dict_from_one_id_to_another_id(from_one_id_to_one_name,
                                       from_one_name_to_another_name,
                                       from_another_name_to_another_id):
    dict = {}
    for one_id in range(len(set(from_one_id_to_one_name.values()))):
        one_name = from_one_id_to_one_name.get(one_id)
        another_name = from_one_name_to_another_name.get(one_name)
        another_id = from_another_name_to_another_id.get(another_name)
        dict[one_id] = another_id
    return dict


# from_coco_id_to_mot_id = get_dict_from_coco_id_to_mot_id()
from_coco_id_to_mot_id = get_dict_from_one_id_to_another_id(
    from_coco_id_to_coco_name,
    from_coco_name_to_mot_name,
    from_mot_name_to_mot_id)

# From VOC to MOT
from_voc_name_to_mot_name = {"BACKGROUND": default_mot,
                             "aeroplane": default_mot,
                             "bicycle": "Bicycle",
                             "bird": default_mot,
                             "boat": default_mot,
                             "bottle": default_mot,
                             "bus": "Car",
                             "car": "Car",
                             "cat": default_mot,
                             "chair": default_mot,
                             "cow": default_mot,
                             "diningtable": default_mot,
                             "dog": default_mot,
                             "horse": default_mot,
                             "motorbike": "Motorbike",
                             "person": "Pedestrian",
                             "pottedplant": default_mot,
                             "sheep": default_mot,
                             "sofa": default_mot,
                             "train": default_mot,
                             "tvmonitor": default_mot}

from_voc_id_to_mot_id = get_dict_from_one_id_to_another_id(
    from_voc_id_to_voc_name, from_voc_name_to_mot_name, from_mot_name_to_mot_id)

mapper = {
    "from_coco_id_to_mot_id": from_coco_id_to_mot_id,
    "from_voc_id_to_mot_id": from_voc_id_to_mot_id
}

if __name__ == "__main__":
    print("coco classes: ", from_coco_name_to_coco_id)

    # check coco person to mot pedestrian
    coco_id = 0
    coco_name = from_coco_name_to_coco_id.get(coco_id)
    mot_name = from_coco_name_to_mot_name.get(coco_name)
    mot_id = from_mot_name_to_mot_id.get(mot_name)
    # mot_id = get_from_coco_id_to_mot_id(coco_id)
    mot_id = from_coco_id_to_mot_id.get(coco_id)
    print("mot_id: ", mot_id)
    assert mot_id == 1

    # check


