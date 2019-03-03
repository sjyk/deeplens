# python -m motmetrics.apps.eval_motchallenge MOT16/train/ res/MOT16/iou_tracker --fmt mot16
import argparse
from motmetrics.apps.eval_motchallenge import eval

def parse_args():
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in 

Milan, Anton, et al. 
"Mot16: A benchmark for multi-object tracking." 
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruths', type=str,
                        default="MOT16/train/",
                        help='Directory containing ground truth files.')
    parser.add_argument('--tests', type=str,
                        default="res/MOT16/iou_tracker/",
                        help='Directory containing tracker result files')
    parser.add_argument('--loglevel', type=str, help='Log level',
                        default='info')
    parser.add_argument('--fmt', type=str, help='Data format',
                        default='mot16')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eval(args)