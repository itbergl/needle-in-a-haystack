"""
    Provides object tracker functionality
"""

import numpy as np

from ..globals import *
from .state_vector import StateVector
from .matrices import Matrices
import cv2
from scipy.optimize import linear_sum_assignment
import pandas as pd
from skimage import measure
import math

SQ_SIZE = {
    'car': 10,
    'plane': 40,
    'ship': 50,
    'train': 60
}

class ObjectTracker():
    def __init__(self, frame_id: int, i: int):
        """
            initialises trackers
        """
        print(f"Initialising tracking of frame {frame_id}")

        self.loci = i
        self.original_im = cv2.cvtColor(g.parser.load_frames(frame_id), cv2.COLOR_BGR2GRAY)
        self.prev_im = cv2.cvtColor(g.parser.load_frames(frame_id-1), cv2.COLOR_BGR2GRAY)
        height, width = self.original_im.shape
        self.COST = ((height**2)+(width**2))**0.5*2/8
        self.COST = 15

        # create matrices object -> allows changing tau
        g.m = Matrices()

        # save the current frame and the hypotheses
        self.current_frame = frame_id

        label = measure.label(g.obj.candidate_masks[self.loci])
        hypothesis = measure.regionprops(label)

        self.current_hypothesis = hypothesis

        # saves a list of all StateVector objects
        self.current_trackers = [StateVector(c.centroid, (SQ_SIZE[g.config.folder]), (SQ_SIZE[g.config.folder])) for c in hypothesis]
        self.half_size = (SQ_SIZE[g.config.folder])/2

        # saves the id of the hypotheses per tracker - init to range
        self.hypotheses_ids = list(range(len(self.current_trackers)))
        self.associate_init()

        self.all_trackers = pd.DataFrame(columns = ["frame_id", "xi", "yi", "xj", "yj", "x", "y"], dtype = np.uint32)

        self.update_trackers()

    def main(self, frame_id: int, i: int):
        print(f"Tracking frame {frame_id}")

        """
            Iterate through another frame
        """
        # save current information
        self.loci = i
        self.current_frame = frame_id
        label = measure.label(g.obj.candidate_masks[self.loci])
        hypothesis = measure.regionprops(label)

        self.current_hypothesis = hypothesis

        # predict new positions: between frames, update track position using the motion model
        for t in self.current_trackers:
            t.update_priori()

        # associate trackers w hypotheses, update trackers, then saves tracked objects for this frame
        self.associate()
        self.update_kalman_filter()
        self.update_trackers()


    def update_trackers(self):
        for cur in self.current_trackers:
            x = np.array(cur.xk[0]).ravel()[0]
            y = np.array(cur.xk[1]).ravel()[0]

            xi = x - cur.width / 2
            xj = x + cur.width / 2
            yi = y - cur.height / 2
            yj = y + cur.height / 2

            df = pd.DataFrame(np.array([[self.current_frame, xi, yi, xj, yj, x, y]]),columns = ["frame_id", "xi", "yi", "xj", "yj", "x", "y"])
            self.all_trackers = self.all_trackers.append(df, ignore_index = True)

    def associate(self):
        """
            associate tracks and hypotheses
        """

        hypothesis = self.current_hypothesis

        hypo_centroids = pd.DataFrame([r.centroid for r in hypothesis], columns = ["x", "y"])
        truth_centroids = [c.prev_hypo_centroid for c in self.current_trackers]

        cost = np.ones((len(hypo_centroids) + len(truth_centroids), len(hypo_centroids) + len(truth_centroids))) * self.COST

        for hypoI in range(len(hypo_centroids)):
            for truthI in range(len(truth_centroids)):
                cost[hypoI, truthI] = math.dist(hypo_centroids.iloc[hypoI], truth_centroids[truthI])

        row_ind, col_ind = linear_sum_assignment(cost)

        unassigned = []

        for cur in self.current_trackers:
            for i in range(len(hypo_centroids)):
                if len(truth_centroids) <= col_ind[i]:
                    continue
                xc = truth_centroids[col_ind[i]][0]
                yc = truth_centroids[col_ind[i]][1]
                cen = (xc, yc)

                if cen == cur.prev_hypo_centroid:
                    if cost[row_ind[i], col_ind[i]] >= self.COST:
                        unassigned.append(cur)
                    xc = hypo_centroids.iloc[i]['x']
                    yc = hypo_centroids.iloc[i]['y']
                    cen = (xc, yc)
                    cur.prev_hypo_centroid = cen

        unassigned = np.array(unassigned)

        # Local searching to remove unmatched tracks
        for i in unassigned:
            prev_pos = np.round(i.prev_hypo_centroid)
            s = np.round(self.half_size)
            y1 = round(prev_pos[1]-s)
            y2 = round(prev_pos[1]+s)
            x1 = round(prev_pos[0]-s)
            x2 = round(prev_pos[0]+s)

            window = self.original_im[y1:y2, x1:x2]
            prev_window = self.prev_im[y1:y2, x1:x2]
            sum_abs_diffs = np.sum(np.abs(np.subtract(window.ravel(), prev_window.ravel(), dtype=np.float)))
            if sum_abs_diffs < self.half_size:
                for j in range(len(self.current_trackers)):
                    if self.current_trackers[j].prev_hypo_centroid == i.prev_hypo_centroid:
                        del self.current_trackers[j]
                        break

    def associate_init(self):
        """
            associate tracks and hypotheses
        """

        hypothesis = self.current_hypothesis

        hypo_centroids = pd.DataFrame([r.centroid for r in hypothesis], columns = ["x", "y"])
        truth_centroids = g.parser.bbox_data.loc[g.parser.bbox_data["frame_id"] == self.loci + 1][["xc", "yc"]]

        cost = np.ones((len(hypo_centroids) + len(truth_centroids), len(hypo_centroids) + len(truth_centroids))) * self.COST

        for hypoI in range(len(hypo_centroids)):
            for truthI in range(len(truth_centroids)):
                cost[hypoI, truthI] = math.dist(hypo_centroids.iloc[hypoI], truth_centroids.iloc[truthI])

        row_ind, col_ind = linear_sum_assignment(cost)

        for cur in self.current_trackers:
            for i in range(len(hypo_centroids)):
                if len(truth_centroids) <= col_ind[i]:
                    continue
                xc = truth_centroids.iloc[col_ind[i]]['xc']
                yc = truth_centroids.iloc[col_ind[i]]['yc']
                cen = (xc, yc)
                if cen == cur.prev_hypo_centroid:
                    xc = hypo_centroids.iloc[i]['x']
                    yc = hypo_centroids.iloc[i]['y']
                    cen = (xc, yc)
                    cur.prev_hypo_centroid = cen

    def update_kalman_filter(self):
        """
            update kalman filter
        """
        for t in self.current_trackers:
            centroids = np.matrix(t.prev_hypo_centroid).getT()
            innovation_k = centroids - g.m.Hk * t.xk
            cov_innovation_k = g.m.Hk * t.Pk * (g.m.Hk.getT()) + g.m.Rk
            kalman_gain_k = t.Pk * (g.m.Hk.getT()) * (cov_innovation_k.getI())
            t.update_posteriori(kalman_gain_k, innovation_k)
