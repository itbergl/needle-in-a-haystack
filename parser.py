"""
    Parser class
"""

# IMPORTS
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import math

from .globals import *

# DEFINED PATHS
BBOX_PATH = os.path.join('gt', 'gt.txt')
CENTROID_PATH = os.path.join('gt', 'gt_centroid.txt')
IM_PATH = 'img'

def list_available_folders(path):
    return sorted([x for x in os.listdir(path) if os.path.isdir(f'{path}/{x}')])

class Parser():
    """
        Parses the required image data and constructs methods for loading required frames
    """
    def __init__(self):
        """
            @param folder: folder name (e.g. 'car')
            @param sequence: sequence number
            @param mot_path: path to mot folder
        """
        frames_path = os.path.join(g.config.mot_path, g.config.folder, g.config.sequence)
        frames = sorted(os.listdir(os.path.join(frames_path, IM_PATH)))

        with open(os.path.join(frames_path, BBOX_PATH), 'r') as file:
            first_line = file.readline().strip()

            delim = ","
            lines = first_line.split(delim)
            if not len(lines) > 1:
                delim = " "

            bbox_data = pd.read_csv(os.path.join(frames_path, BBOX_PATH), header=None, sep=delim, skipinitialspace=True)

            files = []
            for file in frames:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    files.append(os.path.join(frames_path, IM_PATH, file))

            # The '.frame_filenames' attribute contains a list of all frame paths associated with the sequence
            # The '.bbox_data' attribute contains the first 6 columns from the 'gt.txt' file for each sequence
            self.frame_filenames = np.sort(np.array(files))
            self.bbox_data = bbox_data.iloc[:,:6]

            # temp im to get image dimensions
            im = cv2.imread(self.frame_filenames[0])

            # if second columns make up a 1/5th of the width or height, assume coordinates
            if np.max(self.bbox_data[4] / im.shape[1]) > 0.2 or np.max(self.bbox_data[5] / im.shape[0]) > 0.2:
                self.bbox_data.columns = ["frame_id", "track_id", "xi", "yi", "xj", "yj"]

                # add width, height
                self.bbox_data["width"] = self.bbox_data["xj"] + self.bbox_data["xi"]
                self.bbox_data["height"] = self.bbox_data["yj"] + self.bbox_data["yi"]
            else:
                self.bbox_data.columns = ["frame_id", "track_id", "xi", "yi", "width", "height"]
                
                # add xj, yj
                self.bbox_data["xj"] = self.bbox_data["xi"] + self.bbox_data["width"]
                self.bbox_data["yj"] = self.bbox_data["yi"] + self.bbox_data["height"]

            # add centroid data
            self.bbox_data["xc"] = self.bbox_data["xi"] + self.bbox_data["width"] / 2
            self.bbox_data["yc"] = self.bbox_data["yi"] + self.bbox_data["height"] / 2

            self.set_on_demand()

    def check_if_load_batch(self, start_index, final_index):
        """
            Checks whether enough RAM to load all, otherwise loads on demand. Only used
            when detecting and tracking objects

            If enough RAM, creates an array to store files and fills by reading in images.

            Could be optimised to semi-load on-demand ie. load first x files, when need new
            images, load following x after unloading initial x

            @param start_index inc
            @param end_index exc
        """
        # read first img to determine max number of frames
        im = cv2.imread(self.frame_filenames[0])

        # 3/4 as imgs are RGB, and still need to store bin images
        max_frames = int(g.config.max_ram / (im.nbytes / 10**9) * 3/4)
        self.on_demand = not ((final_index - start_index) <= max_frames)

        self.frames = np.empty(len(self.frame_filenames), dtype=np.ndarray)

        if not self.on_demand:
            for i in range(start_index, final_index):
                self.frames[i] = cv2.imread(self.frame_filenames[i], cv2.IMREAD_COLOR)

    def set_on_demand(self):
        """
            Sets to load on demand
        """
        self.on_demand = True
        self.frames = np.empty(len(self.frame_filenames), dtype=np.ndarray)

    def get_frame_count(self):
        """
            Returns the number of frames in this sequence
        """
        return len(self.frame_filenames)

    def load_frames(self, start_index, stop_index = None, rect = False):
        """
            The load_frames method returns the image data for the desired frame(s). If stop_id not passed, only returns 1 frame

            @param start_index:         Starting frame index (starts at 0)
            @param stop_index = None:   Final frame index (exc)
            @param rect = False:        Whether to draw the tracking object bounding boxes
        """
        # no stop index, load only the start_index frame
        if stop_index is None:
            return cv2.imread(self.frame_filenames[start_index], cv2.IMREAD_COLOR) if self.on_demand else self.frames[start_index]

        # stop index given, load many frames
        else:
            ims = []
            for i in range(start_index, stop_index):
                ims.append(cv2.imread(self.frame_filenames[i]) if self.on_demand else self.frames[i])
            return np.array(ims)

    def load_frames_id(self, start_id, stop_id = None, rect = False):
        """
            Wrapper to load frames based on id instead of index

            @param start_id:            Starting frame id (starts at 1)
            @param stop_id = None:      Final frame id (inc)
            @param display = False:     Whether to return a pyplot imshow obj (only applicable for stop_id = None)
            @param rect = False:        Whether to draw the tracking object bounding boxes
        """
        return self.load_frames(start_id - 1, stop_id, rect)