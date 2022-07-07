"""
    Provides functions to evaluate tracking accuracy
"""
from ..globals import *
import numpy as np
import cv2

BBOX_COLS = ["xi", "yi", "xj", "yj"]

def evaluate_tracking(startIndex, finalIndex):
    """
        Evaluate tracking
    """

    # somewhere x and ys got swapped
    g.tracker.all_trackers = g.tracker.all_trackers.rename(columns = {"xi": "yi", "yi": "xi", "xj": "yj", "yj": "xj", "y": "x", "x": "y"})
    g.tracker.all_trackers = g.tracker.all_trackers[["frame_id", "xi", "yi", "xj", "yj"]].astype(np.uint32)

    # refs
    hypos  = g.tracker.all_trackers
    truths = g.parser.bbox_data

    # arrays to store metrics
    precision = np.zeros(finalIndex-startIndex)
    recall = np.zeros(finalIndex-startIndex)
    f1 = np.zeros(finalIndex-startIndex)

    moving_objects = np.zeros(finalIndex - startIndex)
    unmatched = np.zeros(finalIndex - startIndex, dtype = np.float32)
    num_tracks_swapped = np.zeros(finalIndex - startIndex)

    # loop through all frames within span
    for i in range(finalIndex - startIndex):
        print("Evaluating frame " + str(i + startIndex + 1))
        # subset dataframes
        hypos_subset = hypos.loc[hypos["frame_id"] == startIndex+i+1]
        truths_subset = truths.loc[truths["frame_id"] == startIndex+i+1]

        # show image
        show_image(startIndex + i, hypos_subset[BBOX_COLS])

        moving_objects[i] = len(hypos_subset)

        (TP, FP, FN, track_ids_current) = get_tp_fp_fn_ids(hypos_subset, truths_subset)

        # unmatched truths proportion
        unmatched[i] = FN / len(truths_subset)

        # precision and recall
        precision[i] = TP / (TP + FP)
        recall[i] = TP / (TP + FN)
        if recall[i] == 0: f1[i] = 0
        else: f1[i] += 2 * precision[i] * recall[i] / (precision[i] + recall[i])

        # track ids swapped
        if i > 0:
            num_tracks_swapped[i] = np.sum(track_ids_current != track_ids_previous)
        track_ids_previous = track_ids_current

    # fills results with statistics text
    g.gui.display_text("Average Number of Moving Objects: {}\nProp of unmatching Ground Truth Objects: {}\nNumber of Objects that Swapped Tracks: {}\nPrecision: {}\nRecall: {}\nF1: {}\nAverage Precision: {}  -  Average Recall: {}  -  Average F1: {}".format(
        np.array2string(moving_objects, precision = 2),
        np.array2string(unmatched, precision = 2),
        np.array2string(num_tracks_swapped, precision = 2),
        np.array2string(precision, precision = 2),
        np.array2string(recall, precision = 2),
        np.array2string(f1, precision = 2),
        round(np.sum(precision) / len(precision), 4), 
        round(np.sum(recall) / len(recall), 4), 
        round(np.sum(f1) / len(f1), 4)
    ))

    # saves statistics to global variable for save to csv later
    g.tracking_results = "Average Number of Moving Objects,{}\nProp of unmatching Ground Truth Objects,{}\nNumber of Objects that Swapped Tracks,{}\nPrecision,{}\nRecall,{}\nF1,{}\nAverage Precision,{}\nAverage Recall,{}\nAverage F1,{}".format(
        np.array2string(moving_objects, precision = 2, separator = ",")[1:-1],
        np.array2string(unmatched, precision = 2, separator = ",")[1:-1],
        np.array2string(num_tracks_swapped, precision = 2, separator = ",")[1:-1],
        np.array2string(precision, precision = 2, separator = ",")[1:-1],
        np.array2string(recall, precision = 2, separator = ",")[1:-1],
        np.array2string(f1, precision = 2, separator = ",")[1:-1],
        round(np.sum(precision) / len(precision), 4), 
        round(np.sum(recall) / len(recall), 4), 
        round(np.sum(f1) / len(f1), 4)
    )

def get_tp_fp_fn_ids(hypos_subset, truths_subset):
    hypo_to_truth = np.zeros(len(hypos_subset))  - 1 # for each hypos, contains the truth ID
    truth_to_hypo = np.zeros(len(truths_subset)) - 1 # for each truth, contain the ID of the hypos
    track_ids_current = np.zeros(len(hypos_subset))  # track ids of each matched hypo

    # match candidates - loop through hypotheses and truths
    for ip in range(len(hypos_subset)):
        for it in range(len(truths_subset)):
            # calc area
            (A, B, AnB, AuB) = calc_area(hypos_subset.iloc[ip][BBOX_COLS], truths_subset.iloc[it][BBOX_COLS])
            # has some intersection
            if AnB != 0 and hypo_to_truth[ip] == -1:  # and (AnB / AuB) > 0.7 - removed this as our tracking is not accurate
                hypo_to_truth[ip] = it
                truth_to_hypo[it] = ip
                track_ids_current[ip] = truths_subset.iloc[it]["track_id"]

                break
    
    # calculate metrics
    TP = np.sum(hypo_to_truth > -1)
    FP = np.sum(hypo_to_truth == -1)
    FN = np.sum(truth_to_hypo == -1)

    return (TP, FP, FN, track_ids_current)

def calc_area(bbox1, bbox2):
    """
        Given two sets of bounding boxes, calculates:
            area of bbox1 = A
            area of bbox2 = B
            area of intersection = AnB
            area of union = AuB

        Returns (A, B, AnB, AuB)
    """
    xoverlap = (bbox1["xi"] < bbox2["xj"] and bbox1["xj"] > bbox2["xj"]) or (bbox2["xi"] < bbox1["xj"] and bbox2["xj"] > bbox1["xj"])
    yoverlap = (bbox1["yi"] < bbox2["yj"] and bbox1["yj"] > bbox2["yj"]) or (bbox2["yi"] < bbox1["yj"] and bbox2["yj"] > bbox1["yj"])

    if xoverlap and yoverlap:
        # calc area A * B
        A = (bbox1["xj"] - bbox1["xi"]) * (bbox1["yj"] - bbox1["yi"])
        B = (bbox2["xj"] - bbox2["xi"]) * (bbox2["yj"] - bbox2["yi"])

        # calc A intersection B
        xi = bbox1["xi"] if bbox1["xi"] > bbox2["xi"] else bbox2["xi"]
        xj = bbox1["xj"] if bbox1["xj"] < bbox2["xj"] else bbox2["xj"]

        yi = bbox1["yi"] if bbox1["yi"] > bbox2["yi"] else bbox2["yi"]
        yj = bbox1["yj"] if bbox1["yj"] < bbox2["yj"] else bbox2["yj"]

        # calc AnB and AuB
        AnB = (xj - xi) * (yj - yi)
        AuB = A + B - AnB

        return (A, B, AnB, AuB)
    else:
        return (0,0,0,0)

def show_image(i: int, bboxes):
    """
        Draws rectangles for bounding boxes based on trackers and displays the image on the gui
    """
    im = g.parser.load_frames(i, i+1)[0]
    g.gui.display_image(im, bboxes[BBOX_COLS], frame_id=i+1)