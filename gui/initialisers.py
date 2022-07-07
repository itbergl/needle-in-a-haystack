from ..globals import *
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (QFileDialog)
import cv2
from PIL import Image

"""
    Max RAM
"""
def max_ram(gui):
    gui.elements["max_ram"].setValidator(QIntValidator(1, 64))
    gui.elements["max_ram"].setText("2")
    gui.elements["max_ram"].setMaximumWidth(64)
    g.config.max_ram = 2

    gui.elements["max_ram"].textChanged.connect(ram_changed)

def ram_changed(value):
    if value:
        g.config.max_ram = int(value)

"""
    Changing hyperparameters
"""
def sigma(gui):
    for (i, widg_id) in enumerate(["sigma_p", "sigma_v", "sigma_a", "sigma_d"]):
        gui.elements[widg_id].setValidator(QDoubleValidator(0, 10, 3))
        gui.elements[widg_id].setText("1.0")
        gui.elements[widg_id].setMaximumWidth(64)

        # defaults already set
        change_funcs = {0: sigma_p, 1: sigma_v, 2: sigma_a, 3: sigma_d}
        gui.elements[widg_id].textChanged.connect(change_funcs[i])

def sigma_p(value):
    if value:
        g.hyperparams.sigmas[0] = float(value)
def sigma_v(value):
    if value:
        g.hyperparams.sigmas[1] = float(value)
def sigma_a(value):
    if value:
        g.hyperparams.sigmas[2] = float(value)
def sigma_d(value):
    if value:
        g.hyperparams.sigmas[3] = float(value)

def tau(gui):
    gui.elements["tau"].setValidator(QIntValidator(1, 10))
    gui.elements["tau"].setText("1")
    gui.elements["tau"].setMaximumWidth(64)

    gui.elements["tau"].textChanged.connect(tau_changed)

def tau_changed(value):
    if value:
        g.hyperparams.tau = int(value)

"""
    Save results
"""
def save_results(gui):
    gui.elements["save_results"].clicked.connect(save_results_clicked)

def save_results_clicked(self):
    if g.tracked:
        path = QFileDialog.getSaveFileName(g.gui, 'Choose a file to save the results to.')
        if path: 
            with open(f"{path[0]}.txt", "w") as f:
                f.write(g.tracking_results) 

"""
    save gif of tracking
"""
def save_gif(gui):
    gui.elements["save_gif"].clicked.connect(save_gif_clicked)

def save_gif_clicked(self):
    """
        Save a GIF of the tracked images
    """
    if g.tracked:
        # get the user to give a file path
        path = QFileDialog.getSaveFileName(g.gui, 'Choose a file to save the GIF to.')[0]
        if path: 
            images = g.parser.load_frames_id(g.config.start_id, g.config.final_id)

            width = images.shape[2]
            height = images.shape[1]

            # for each of the images, draw bounding boxes of hypotheses
            for i in range(g.config.start_id, g.config.final_id+1):
                for index, box in g.tracker.all_trackers.loc[g.tracker.all_trackers["frame_id"] == i][["xi", "yi", "xj", "yj"]].iterrows():
                    if box[0] > width or box[2] > width or box[1] > height or box[3] > height: continue
                    else:
                        cv2.rectangle(images[i - g.config.start_id], (box["xi"], box["yi"]), (box["xj"], box["yj"]), color = (255,0,255))

            # convert all images to PIL objects
            images = [Image.fromarray(im).convert('P') for im in images]

            # if the path doesn't end in '.gif', add '.gif', and save
            if path[-4:] != ".gif": path = path + ".gif"

            images[0].save(path, save_all=True, append_images=images[1:], duration=100)