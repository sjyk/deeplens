# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
#
# Updated by Adam Dziedzic (stateful IOU tracker with label checking).
# ---------------------------------------------------------

from time import time
from object_tracking.interface_tracker import Tracker
from object_tracking.iou_tracker.util import load_mot, iou
from deeplens_utils.class_mapper import from_mot_name_to_mot_id, from_mot_id_to_mot_name

person_label = [from_mot_name_to_mot_id.get("Pedestrian")]
all_mot_labels = from_mot_id_to_mot_name.keys()

class IouTracker(Tracker):
    def __init__(self, sigma_l=0.3, sigma_h=0.5, sigma_iou=0.3, t_min=5,
                 t_max=2, match_labels="no"):
        """
        Stateful IOU (Intersection over Union) tracker.

        :param sigma_l: (float) low detection threshold.
        :param sigma_h: (float) high detection threshold.
        :param sigma_iou: (float) IOU threshold.
        :param t_min: (int) minimum track length in frames.
        :param t_max: (int) maximum number of frames during which a track can be
        suspended.
        """
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min
        self.t_max = t_max
        self.match_labels = False if match_labels == "no" else True

        self.reset()

    def get_params(self):
        return ["sigma_l", self.sigma_l, "sigma_h", self.sigma_h, "sigma_iou",
                self.sigma_iou, "t_min", self.t_min, "t_max", self.t_max,
                "match_labels", self.match_labels]

    def reset(self):
        """
        Reset the state of the tracker.
        """
        self.active_tracks = []
        self.suspended_tracks = []
        self.frame_num = 0
        self.obj_id = 0

    def update(self, dets, class_labels=frozenset(all_mot_labels)):
        """
        Update the tracks with the new detections for a new frame. Make the
        iou tracker work in an online fashion (work on a frame at a time basis).
        Every frame should be provided, even if there are no detections (because
        we have to update the finished and suspended tracks).

        :param dets: list of detections for a frame
        dets are in the format specified in
        :class:`~object_detection.interface_decorator`
        deeplens/object_detection/interface_detector.py
        :class_labels: which labels should be considered for the detections and
        tracks. By default, we only consider people (pedestrians).
        :return: all current tracks
        """
        self.frame_num += 1

        score_idx = 4
        label_idx = 5

        if len(dets) == 0:
            dets = []
        else:
            # apply low threshold to detections
            # dets = dets[(dets[:,score_idx] >= self.sigma_l).nonzero().squeeze(1)]
            dets = dets[dets[:, score_idx] >= self.sigma_l]

            bbox = dets[:,0:score_idx]
            scores = dets[:,score_idx]
            labels = dets[:,label_idx]

            # Reformat the detections: divide the detections into 3 sections: bbox,
            # score, and label.
            dets = []

            for bb, s, label in zip(bbox, scores, labels):
                if label in class_labels:  # only consider pedestrians by default
                    dets.append(
                        {'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s,
                         'label': int(label)})

        updated_tracks = []
        finished_tracks = []
        # Try to find matches for the suspended_tracks.
        self.active_tracks += self.suspended_tracks
        # remove tracks that suspended too long, i.e. longer than t_max
        self.suspended_tracks = []
        for track in self.active_tracks:
            if len(dets) > 0:
                # Get det with the same label and the highest iou for the track.
                # Consider only the last track for all the tracks collected in
                # this "active track".
                # Main change: take only the dets with the same label as the
                # active track.
                # dets_label = dets[dets[:, label_idx] == track['label']]
                if self.match_labels:
                    dets_label = [det for det in dets if
                                  det['label'] == track['label']]
                else:
                    dets_label = dets

                last_track = track['bboxes'][-1]
                best_match = max(dets_label,
                                 key=lambda x: iou(last_track, x['bbox']))
                if iou(last_track, best_match['bbox']) >= self.sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['scores'].append(best_match['score'])
                    track['max_score'] = max(track['max_score'],
                                             best_match['score'])
                    # the track label should not change
                    # assert track['label'] == best_match['label']
                    track['frames'].append(self.frame_num)

                    updated_tracks.append(track)

                    # Remove the best matching detection from detections.
                    del dets[dets.index(best_match)]

            # if there are not detections or the track was not updated
            if len(dets) == 0 or track is not updated_tracks[-1]:
                """
                Finish tracks that are with max_score >= sigma_h, already long 
                enough >= t_min, and did not re-emerge after t_max frames.
                """
                track_last_frame = track['frames'][-1]
                if track['max_score'] >= self.sigma_h and len(
                        track['bboxes']) >= self.t_min and (
                        self.frame_num - track_last_frame) > self.t_max:
                    finished_tracks.append(track)
                elif (self.frame_num - track_last_frame) <= self.t_max:
                    # If the track was not suspended for longer then t_max then
                    # keep it suspended. Otherwise, it should be discarded.
                    self.suspended_tracks.append(track)

        # Create new tracks.
        new_tracks = []
        for det in dets:
            self.obj_id += 1  # MOT requires obj_id >= 1
            new_tracks.append({
                'obj_id': self.obj_id,
                'bboxes': [det['bbox']],  # bounding boxes for the track
                'scores': [det['score']],
                # scores for each frame of the track
                'max_score': det['score'],
                # max score for the whole track
                'label': det['label'],
                # constant label for the whole track
                'frames': [self.frame_num]
                # frame numbers where the object appeared
            })

        self.active_tracks = updated_tracks + new_tracks

        return self.active_tracks, finished_tracks

    def track(self, dets):
        active_tracks, _ = self.update(dets)
        return_tracks = []
        for active_track in active_tracks:
            obj_id = active_track['obj_id']
            bbox = active_track['bboxes'][-1]  # the last bounding box
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            score = active_track['scores'][-1]
            class_id = active_track['label']
            return_tracks.append([obj_id, x1, y1, x2, y2, score, class_id])
        return return_tracks

def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information
    by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.

    Args:
         detections (list): list of detections per frame, usually generated by
         util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        list: list of tracks.
    """

    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1],
                                                         x['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'],
                                             best_match['score'])
                    track['label'] = best_match['label']

                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(
                        track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'],
                       'label': det['label'],
                       'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(
            track['bboxes']) >= t_min]

    return tracks_finished


def track_iou_matlab_wrapper(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Matlab wrapper of the iou tracker for the detrac evaluation toolkit.

    Args:
         detections (numpy.array): numpy array of detections, usually supplied by run_tracker.m
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        float: speed in frames per second.
        list: list of tracks.
    """

    detections = detections.reshape((7, -1)).transpose()
    dets = load_mot(detections)
    start = time()
    tracks = track_iou(dets, sigma_l, sigma_h, sigma_iou, t_min)
    end = time()

    id_ = 1
    out = []
    for track in tracks:
        for i, bbox in enumerate(track['bboxes']):
            out += [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]),
                    float(bbox[3] - bbox[1]),
                    float(track['start_frame'] + i), float(id_)]
        id_ += 1

    num_frames = len(dets)
    speed = num_frames / (end - start)

    return speed, out
