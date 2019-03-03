"""
Count number of objects in the ground truth tracks vs. the number of cars from
the detected tracks via iou (intersection over union) method.
"""

import pandas as pd
import argparse
import os

# General names in the files and path to the detection files.
default_header = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'class',
                  'visibility_ratio', 'wz']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Queries on the outputs from MOT benchmark.")
    parser.add_argument('--label_id', type=int, default=1)
    parser.add_argument('--dataset', default="MOT16-02")
    parser.add_argument('--tracks_version', default="_yolo")
    return parser.parse_args()


def run_query(label_id, dataset):
    print("detected people (sample): \n",
          dataset[dataset['class'] == label_id].head(5))
    print("number of detected people (in all frames separately): ",
          dataset[dataset['class'] == label_id].size)
    print("number of frames with a distinctly tracked people: \n",
          dataset[dataset['class'] == label_id].groupby(
              ["id", "class"]).size())
    """
    We take into account only entries with a person class label. Then we count 
    the unique person entries.
    """
    print("number of distinct pedestrians in the whole video: ",
          dataset[dataset['class'] == label_id].groupby(
              ["id", "class"]).size().size)


def count_all_people(dataset):
    # Count also the entries without a label (by default we assume there are
    # people.
    return dataset[(dataset['class'] == -1) | (dataset['class'] == 1) | (
            dataset['class'] == 2) | (
                           dataset['class'] == 7)].groupby(
        ["id", "class"]).size().size


def count_all_objects(dataset):
    # Count also the entries without a label (by default we assume there are
    # people.
    return dataset.groupby(["id", "class"]).size().size


def find_all_people(dataset):
    print(
        f"{dataset.name}: number of distinct people (pedestrian, person on "
        f"vehicle, or a static person) in the whole video: ",
        count_all_people(dataset=dataset)
    )


def find_people_per_frame(dataset):
    print("number of people tracked per frame: ",
          dataset[(dataset['class'] == -1) | (dataset['class'] == 1) | (
                  dataset['class'] == 2) | (
                          dataset['class'] == 7)].groupby("frame").size())


def find_avg_num_people_per_frame(dataset):
    print(f"{dataset.name}: avg number of people tracked per frame: ",
          count_avg_num_people_per_frame(dataset=dataset))


def count_avg_num_people_per_frame(dataset):
    return dataset[(dataset['class'] == -1) | (dataset['class'] == 1) | (
            dataset['class'] == 2) | (
                           dataset['class'] == 7)].groupby(
        "frame").size().mean()


def count_avg_num_objects_per_frame(dataset):
    return dataset.groupby("frame").size().mean()


def show_one_by_one(main_seq, args):
    ground_truth_file = "../MOT16/train/" + main_seq + "/gt/gt.txt"
    ground_truth = pd.read_csv(ground_truth_file, sep=",", header=None,
                               names=["frame", "id", "x", "y", "w",
                                      "h", "score", "class",
                                      "visibility_ratio"])
    ground_truth.name = "ground-truth"
    print(ground_truth.head(5))

    label_id = args.label_id

    run_query(label_id=label_id, dataset=ground_truth)
    find_all_people(dataset=ground_truth)
    find_people_per_frame(dataset=ground_truth)
    find_avg_num_people_per_frame(dataset=ground_truth)

    iou_tracks_file = "../res/MOT16/iou_tracker/" + main_seq + args.tracks_version + ".txt"
    print("iou_tracks_file: ", iou_tracks_file)
    iou_tracks = pd.read_csv(iou_tracks_file, sep=",", header=None,
                             names=['frame', 'id', 'x', 'y', 'w', 'h', 'score',
                                    'class', 'wy', 'wz'])
    iou_tracks.name = "iou-tracks (fast MOT)"
    print(iou_tracks.head(5))
    run_query(label_id=label_id, dataset=iou_tracks)
    find_all_people(dataset=iou_tracks)
    find_people_per_frame(dataset=iou_tracks)
    find_avg_num_people_per_frame(dataset=iou_tracks)

    yolo_sort_file = "../../motchallenge/MOT16/train/" + main_seq + "/det_yolo_sort/det.txt"
    yolo_sort = pd.read_csv(yolo_sort_file, sep=",", header=None,
                            names=['frame', 'id', 'x', 'y', 'w', 'h', 'score',
                                   'class', 'wy', 'wz'])
    yolo_sort.name = "yolo_sort"
    print(yolo_sort.head(5))
    run_query(label_id=label_id, dataset=yolo_sort)
    find_all_people(dataset=yolo_sort)
    find_people_per_frame(dataset=yolo_sort)
    find_avg_num_people_per_frame(dataset=yolo_sort)


def show_statistics(dataset_path, name=None):
    if name is None:
        name = dataset_path.split("/")[-1]
    df = pd.read_csv(dataset_path, sep=",", header=None, names=default_header)
    df.name = name
    # pd.set_option('display.max_columns', None)
    # run_query(label_id=args.label_id, dataset=df)
    find_all_people(dataset=df)
    # find_people_per_frame(dataset=df)
    find_avg_num_people_per_frame(dataset=df)


def get_statistics(dataset_path, name=None):
    if name is None:
        name = dataset_path.split("/")[-1]
    df = pd.read_csv(dataset_path, sep=",", header=None, names=default_header)
    df.name = name
    # pd.set_option('display.max_columns', None)
    # run_query(label_id=args.label_id, dataset=df)
    distinct_people = count_all_people(dataset=df)
    # find_people_per_frame(dataset=df)
    avg_people_frame = count_avg_num_people_per_frame(dataset=df)
    return ["distinct_people", distinct_people, "avg_people_per_frame",
            avg_people_frame]


def main(args):
    main_seq = args.dataset
    # print("main_seq: ", main_seq)
    # show_one_by_one(main_seq, args)

    # The path to the ground truth files from the MOT16 benchmark.
    ground_truth_path = "../../motchallenge/MOT16/train/" + main_seq + "/gt/"
    ground_truth_header = ["frame", "id", "x", "y", "w", "h", "score", "class",
                           "visibility_ratio"]

    # print("current path: ", os.getcwd())
    # default_path = "../../motchallenge/MOT16/train/" + main_seq + "/det/"
    default_path = "../../motchallenge/res/MOT16/iou_tracker_from_gt_det/" + main_seq + ".txt"

    # for dataset in ["ground_truth", "iou_tracker", "yolo_sort",
    #                 "vgg16-ssd_top_k_20_filter_threshold_0.1_candidate_size_200_nms_method_soft"]:
    # for dataset in ["ground_truth"]:
    for dataset in ["iou_tracker_from_gt_det"]:
        # full_path = os.path.join(default_path, "det_" + dataset + ".txt")
        full_path = default_path
        # print("full path: ", full_path)
        header_names = default_header
        if dataset == "ground_truth":
            full_path = os.path.join(ground_truth_path, "gt.txt")
            header_names = ground_truth_header
        df = pd.read_csv(full_path, sep=",", header=None, names=header_names)
        df.name = dataset
        pd.set_option('display.max_columns', None)
        # print(dataset + ": \n", df.head(5))
        # run_query(label_id=args.label_id, dataset=df)
        # print(main_seq)
        # find_all_people(dataset=df)
        count = count_all_people(dataset=df)
        # find_people_per_frame(dataset=df)
        # find_avg_num_people_per_frame(dataset=df)
        avg = count_avg_num_people_per_frame(dataset=df)
        count_obj = count_all_objects(dataset=df)
        avg_obj = count_avg_num_objects_per_frame(dataset=df)
        print(main_seq, count, avg, count_obj, avg_obj)


if __name__ == '__main__':
    args = parse_args()
    print("bench_case", "#_people", "#_people_per_frame", "#_objects", "#_objects_per_frame")
    for dataset in ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10",
                    "MOT16-11", "MOT16-13"]:
        args.dataset = dataset
        main(args)
