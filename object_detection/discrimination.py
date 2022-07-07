"""
    Provides candidate discrimination functionality
"""

import math
import numpy as np
import pandas as pd
import cv2
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from skimage import measure
import matplotlib.pyplot as plt

from ..globals import *
#from ..evaluate import evaluate_discrimination
from .detection import CandidateDetection

# Set constants
P_NORM = 0.995
CUES = {'area': [36, 625],
        'extent': [0.8, 1],
        'axis_major_length': [11, 30],
        'eccentricity': [0, 0.8]
        }

SQ_SIZE = {
    'car': 10,
    'plane': 40,
    'ship': 50,
    'train': 60
}

class CandidateDiscrimination():
    # CANDIDATE DISCRIMINATION
    def discriminate_candidates(self, i: int, loci):
        """
            Given an image index i, perform discrimination
        """

        self.original_im = cv2.cvtColor(g.parser.load_frames(i), cv2.COLOR_BGR2GRAY)
        height, width = self.original_im.shape
        self.COST = ((height**2)+(width**2))**0.5*2

        self.loci = loci
        self.region_growing()
        self.morph_cues()

        # Calibration not required for functionality as the values have been set
        # self.calibrate(i)


    def region_growing(self):
        """
            Given an image index i, perform region growing
        """
        # Determining individual candidate pixel clusters (regions) from binary candidate image
        label = measure.label(g.obj.candidate_masks[self.loci])
        regions = measure.regionprops(label)

        # Iterating through each candidate cluster for region growing
        for region in regions:
            pixel_value = 255
            if region.area < 2:
                pixel_value = 0
            x, y = region.centroid[0], region.centroid[1] # centroid coordinates, row, col
            x = round(x)
            y = round(y)
            orig_grown_region = self.original_im[x-5:x+6, y-5:y+6] # 11 by 11 region from original greyscale im
            region_pixel_values = np.array([self.original_im[c[0], c[1]] for c in region.coords])

            # Generating region growing mask for values within specified margin of error
            mean = np.mean(region_pixel_values)
            stdev = np.std(region_pixel_values)
            ppf = norm.ppf(1-P_NORM, loc=mean, scale=stdev)

            interval_value = stdev * ppf
            lower = mean - interval_value
            upper = mean + interval_value
            growing_mask = (orig_grown_region<upper)&(orig_grown_region>lower)

            # Updating candidate image with updated candidate cluster
            new_candidate = np.zeros(orig_grown_region.shape)
            new_candidate[growing_mask] = pixel_value
            g.obj.candidate_masks[self.loci][x-5:x+6, y-5:y+6] = new_candidate


    def morph_cues(self):
        """
            Perform morphological cue discrimination and return centroids and bounding boxes
        """
        # Determining individual candidate pixel clusters (regions) from binary candidate image
        label = measure.label(g.obj.candidate_masks[self.loci])
        regions = measure.regionprops(label)

        # Iterating through each candidate cluster for morphological discrimination
        for region in regions:
            for cue in CUES:
                cue_value = region[cue]
                if not (CUES[cue][0] <= cue_value <= CUES[cue][1]):
                    for coords in region.coords:
                        g.obj.candidate_masks[self.loci][coords[0], coords[1]] = 0
                    break

        # Returning final bounding boxes and centroids for remaining candidates
        label = measure.label(g.obj.candidate_masks[self.loci])
        final_regions = measure.regionprops(label)
        self.bbox = [r.bbox for r in final_regions]
        self.centroids = [r.centroid for r in final_regions]

    def calibrate(self, i):
        """
            Perform calibration to determine best thresholds for morphological cues
        """
        label = measure.label(g.obj.candidate_masks[self.loci])
        hypothesis = measure.regionprops(label)

        hypo_centroids = pd.DataFrame([r.centroid for r in hypothesis], columns = ["x", "y"])
        truth_centroids = g.parser.bbox_data.loc[g.parser.bbox_data["frame_id"] == i + 1][["xc", "yc"]]

        # allows hypo and truth to both be unmatched w a default cost of COST
        cost = np.ones((len(hypo_centroids) + len(truth_centroids), len(hypo_centroids) + len(truth_centroids))) * self.COST

        for hypoI in range(len(hypo_centroids)):
            for truthI in range(len(truth_centroids)):
                cost[hypoI, truthI] = math.dist(hypo_centroids.iloc[hypoI], truth_centroids.iloc[truthI])

        # Hungarian method to associate tracks
        row_ind, col_ind = linear_sum_assignment(cost)

        truth_values = []
        for i in range(len(row_ind)):
            if cost[row_ind[i], col_ind[i]] == self.COST:
                continue
            for c in range(len(hypo_centroids)): 
                centroid = (hypo_centroids.iloc[c]['x'], hypo_centroids.iloc[c]['y'])
                for hypo in hypothesis:
                    if hypo.centroid == centroid:
                        truth_values.append(hypo.area)

         # Plotting curves for calibration

        plt.plot(truth_values)
        plt.show()

