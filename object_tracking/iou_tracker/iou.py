# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
#
# Updated by Adam Dziedzic (stateful IOU tracker with label checking).
# ---------------------------------------------------------

from time import time

from .util import load_mot, iou

person_label = 1

class IouTracker():
    def __init__(self, sigma_l, sigma_h, sigma_iou, t_min, t_max=3):
        """
        Stateful IOU (Intersection over Union) tracker.

        :param sigma_l: (float) low detection threshold.
        :param sigma_h: (float) high detection threshold.
        :param sigma_iou: (float) IOU threshold.
        :param t_min: (int) minimum track length in frames.
        :param t_max: (int) maximum number of frames during which a track can be
        suspended.
        """
        self.simga_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min

        self.reset()

    def reset(self):
        """
        Reset the state of the tracker.
        """
        self.tracks_active = []
        self.frame_count = 0

    def update(self, dets, class_labels=frozenset([person_label])):
        """
        Update the tracks with the new detections for a new frame. Make the
        iou tracker work in an online fashion (work on a frame at a time basis).

        :param dets: list of detections for a frame
        dets are in the format specified in
        :class:`~object_detection.interface_decorator`
        deeplens/object_detection/interface_detector.py
        :class_labels: which labels should be considered for the detections and
        tracks. By default, we only consider people (pedestrians).
        :return: all current tracks
        """
        self.frame_count += 1

        score_idx = 4
        label_idx = 5

        # apply low threshold to detections
        # dets = dets[(dets[:,score_idx] >= self.sigma_l).nonzero().squeeze(1)]
        dets = dets[dets[:,score_idx] >= self.sigma_l]

        bbox = dets[0:score_idx]
        scores = dets[score_idx]
        labels = dets[label_idx]

        # Reformat the detections: divide the detections into 3 sections: bbox,
        # score, and label.
        dets = []
        for bb, s, label in zip(bbox, scores, labels):
            if label in class_labels:  # only consider pedestrians by default
                dets.append(
                    {'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s,
                     'label': int(label)})

        updated_tracks = []
        suspended_tracks = []
        for track in self.tracks_active:
            if len(dets) > 0:
                # Get det with the same label and the highest iou for the track.
                # Consider only the last track for all the tracks collected in
                # this "active track".
                # Main change: take only the dets with the same label as the
                # active track.
                dets_label = dets[dets[:,label_idx] == track['label']]

                last_track = track['bboxes'][-1]
                best_match = max(dets_label,
                                 key=lambda x: iou(last_track, x['bbox']))
                if iou(last_track, best_match['bbox']) >= self.sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'],
                                             best_match['score'])
                    # the track label should not change
                    assert track['label'] == best_match['label']

                    updated_tracks.append(track)

                    # Remove the best matching detection from detections.
                    del dets[dets.index(best_match)]

            # if track was not updated
            if track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= self.sigma_h and len(
                        track['bboxes']) >= self.t_min and :
                    tracks_finished.append(track)

        # Create new tracks.
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'],
                       'label': det['label'],
                       'start_frame': self.frame_count,
                       'end_frame': self.frame_count} for det in dets]
        self.tracks_active = updated_tracks + new_tracks

        # finish all remaining active tracks
        tracks_finished += [track for track in self.tracks_active
                            if track['max_score'] >= self.sigma_h and len(
                track['bboxes']) >= self.t_min]

        return tracks_finished

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
        suspended_tracks = []
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
                        track['bboxes']) >= t_min and (frame_num - track['end_frame']):
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
