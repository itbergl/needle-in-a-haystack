"""
    Provides object detection functionality
"""

import numpy as np
import cv2
import math

from ..parser import Parser
from ..globals import *

import matplotlib.pyplot as plt

LOCAL_SIZE = (30, 30)
P_FA = 0.05

class CandidateDetection():
    # DETECT CANDIDATES
    def detect_candidates(self, ind: int, loci):
        """
            Given a frame number, returns a binary image of that frame containing candidate objects (=255)

            @param ind: frame index
        """
        # load im and convert to greyscale
        images = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.int16) for im in g.parser.load_frames(ind-1, ind+2)]

        # create template output image with all zeros
        g.obj.candidate_masks[loci] = np.zeros_like(images[1])

        # split into regions
        for j in range(0, images[0].shape[0], LOCAL_SIZE[1]): # y
            for i in range(0, images[0].shape[1], LOCAL_SIZE[0]): # x
                # take differences and threshold
                diffs = self.inter_frame_differences(images, i, j)
                diffs = self.thresholding(diffs)

                # candidate extraction
                g.obj.candidate_masks[loci] = self.candidate_extraction(g.obj.candidate_masks[loci], diffs, i, j)


    def inter_frame_differences(self, images, i: int, j: int):
        """
            Given i and j, sets self.diffs to the absolute differences of the frames

            @param i
            @param j
        """
        # take subset of size LOCAL_SIZE then take differences
        im_subsets = [im[j:j+LOCAL_SIZE[1], i:i+LOCAL_SIZE[0]] for im in images]

        diffs = [
            np.abs(im_subsets[0] - im_subsets[1]).astype(np.uint8), # reconvert to uint8 [0:255] from int16
            np.abs(im_subsets[1] - im_subsets[2]).astype(np.uint8)
        ]

        return diffs

    def thresholding(self, diffs):
        """
            Fits an exponential distribution to the absolute differences to find a threshold.
            Applies the thresholds to the abs differences to create a binary mask.
        """
        for a in [0, 1]:
            # simplified th calculation
            # taking mean of both interframe differences as it was more effective
            th = -1.0 * math.log(P_FA) * np.mean(diffs)
            ret, diffs[a] = cv2.threshold(diffs[a], th, 255, cv2.THRESH_BINARY)

        return diffs

    def candidate_extraction(self, output_image, diffs, i: int, j: int):
        """
            Given i and j, combines thresholded differences and sets the output image to white

            @param i
            @param j
        """
        # combine thresholded masks
        mask = diffs[0] & diffs[1]

        # isolate x and y coords
        ys, xs = np.where(mask == 255)

        # if some detected, add offsets, then set pos to 255
        if len(xs) > 0 and len(ys) > 0:
            xs += i
            ys += j

            output_image[ys, xs] = 255

        return output_image

