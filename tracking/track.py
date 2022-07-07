
import os
import numpy as np
from ..globals import *
from .object_tracker import ObjectTracker
from .evaluate import evaluate_tracking
import datetime


def track():
    """
        Perform tracking
    """
    # check valid start and final ids
    if g.config.start_id >= g.config.final_id or g.tracked: return

    # create index variables
    start_index = g.config.start_id - 1
    final_index = g.config.final_id - 1 + 1

    # checks if the parser can load all images, or whether to query on demand
    g.gui.display_text("Loading files...")
    g.parser.check_if_load_batch(start_index-1, final_index+1) # load a frame before and a frame after for candidate det

    # candidate masks
    g.gui.display_text("Detecting candidates...")
    make_candidate_masks(start_index, final_index)

    # track
    g.gui.display_text("Tracking...")
    g.tracker = ObjectTracker(start_index + 1, 0) #init
    [g.tracker.main(start_index + i + 1, i) for i in range(1, final_index - start_index)] # iter

    # evaluate
    g.gui.display_text("Evaluating...")
    evaluate_tracking(start_index, final_index)

    # sets parser to go back to on demand loading
    g.parser.set_on_demand()

    g.tracked = True

def make_candidate_masks(start_index: int, final_index: int):
    # find candidates
    g.obj.candidate_masks = np.empty(final_index - start_index, dtype = np.ndarray)

    # loop through images finding candidates
    for ind in range(final_index - start_index):
        g.obj.det.detect_candidates(start_index + ind, ind)
        g.obj.disc.discriminate_candidates(start_index + ind, ind)
