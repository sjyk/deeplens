"""
Count number of cars in the ground truth tracks vs. the number of cars from the
detected tracks via iou (intersection over union) method.
"""

import pandas as pd

main_seq = "MOT16-13"
print("main_seq: ", main_seq)

ground_truth_file = "../MOT16/train/" + main_seq + "/gt/gt.txt"
ground_truth = pd.read_csv(ground_truth_file, sep=",", header=None,
                           names=["frame", "id", "x", "y", "w",
                                  "h", "score", "class",
                                  "visibility_ratio"])
print(ground_truth.head(5))

print("detected cars (sample): \n",
      ground_truth[ground_truth['class'] == 3].head(5))
print("number of detected cars (in all frames separately): ",
      ground_truth[ground_truth['class'] == 3].size)
print("number of frames with a distinctly tracked car: \n",
      ground_truth[ground_truth['class'] == 3].groupby(["id", "class"]).size())
print("number of distinct cars in the whole video: ",
      ground_truth[ground_truth['class'] == 3].groupby(
          ["id", "class"]).size().size)

iou_tracks_file = "../res/MOT16/iou_tracker/" + main_seq + ".txt"
iou_tracks = pd.read_csv(iou_tracks_file, sep=",", header=None,
                         names=['frame', 'id', 'x', 'y', 'w', 'h', 'score',
                                'wx', 'wy', 'wz'])
print(iou_tracks.head(5))
