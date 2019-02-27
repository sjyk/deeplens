other = 8
mot_classes = {"person": 1, "car": 3, "bicycle": 4, "motorbike": 5,
               "other": other, "track": 3, "bus": 3}


def get_mot_class_id(class_name):
    return mot_classes.get(class_name, other)
