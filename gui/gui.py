"""
    Creates a GUI class that handles all user interactions with the GUI.
"""

# LIBRARIES
import os
import random
import cv2
import numpy as np

# PyQT Image libs
from PIL.ImageQt import ImageQt
from PIL import Image

# PyQT libs
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QFont, QIntValidator, QPen, QPainter
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton, QGridLayout,
                               QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QFileDialog)

# custom libs
from ..parser import Parser, list_available_folders
from ..tracking.track import track

# global var
from ..globals import *
from .gui_outline import OUTLINE

def convertCvImage2QtImage(cv_img):
    """
        Converts the given opencv image (np array) to QPixmap
        Ref: https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
    """
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(rgb_image).convert('RGB')
    return QPixmap.fromImage(ImageQt(PIL_image))

class GUI(QWidget):
    """
        GUI class
    """
    def __init__(self):
        QWidget.__init__(self)

        # Layouts
        grid = QGridLayout(self)

        # dict to store all elements of the grid
        self.elements = {}

        # title font
        self.title_font = QFont()
        self.title_font.setBold(True)

        g.gui = self

        # define Widgets
        for cat_k, cat in OUTLINE.items():
            for widg_id, widg in cat.items():
                self.elements[widg_id] = widg.construct()
                grid.addWidget(self.elements[widg_id], *widg.pos)

                if widg.Qtype == QLabel:
                    self.elements[widg_id].setText(widg.name)
                
                if "title" in widg_id or widg_id == "image":
                    self.elements[widg_id].setStyleSheet("border: 1px solid black;")
                    self.elements[widg_id].setAlignment(Qt.AlignCenter)
                    self.elements[widg_id].setFont(self.title_font)

                if widg_id == "results":
                    self.elements[widg_id].setAlignment(Qt.AlignTop)


                if widg.init is not None:
                    widg.init(self)

        # check valid dir, else get one
        if not os.path.isdir(g.config.mot_path):
            self.choose_mot_path()

        self.update_elements()
        self.connect_signals()

        self.img = None

    def connect_signals(self):
        # Connecting signals
        self.elements["folder"].currentTextChanged.connect(self.on_folder_changed)
        self.elements["sequence"].currentTextChanged.connect(self.on_sequence_changed)
        self.elements["display_frame_id"].currentTextChanged.connect(self.on_frame_changed)
        self.elements["start_frame_id"].currentTextChanged.connect(self.on_start_id_changed)
        self.elements["end_frame_id"].currentTextChanged.connect(self.on_final_id_changed)

        # clicks
        self.elements["display"].clicked.connect(self.on_display_click)
        self.elements["clear_display"].clicked.connect(self.clear_display)
        self.elements["mot_path"].clicked.connect(self.choose_mot_path)
        self.elements["track"].clicked.connect(track)

    def update_elements(self):
        # Initialisers
        self.elements["folder"].clear()
        self.elements["folder"].addItems(list_available_folders(g.config.mot_path))
        g.config.folder = self.elements["folder"].currentText()

        self.update_sequence()
        g.config.sequence = self.elements["sequence"].currentText()

        self.update_frame()
        g.config.frame = int(self.elements["display_frame_id"].currentText())

        self.elements["image"].setText("No image loaded.")

        g.config.start_id = int(self.elements["start_frame_id"].currentText())
        g.config.final_id = int(self.elements["end_frame_id"].currentText())

        g.parser = Parser()

    """
        Input values changing
    """
    def values_changed(f):
        def wrapper(self, *args, **kwargs):
                f(self, *args, **kwargs)
                self.elements["image"].clear()
                self.img = None
        return wrapper

    @values_changed
    def on_folder_changed(self, value):
        g.config.folder = value
        self.update_sequence()
        self.update_frame()

    @values_changed
    def on_sequence_changed(self, value):
        if value:
            g.config.sequence = value
            g.parser = Parser()
            self.update_frame()

    @values_changed
    def on_frame_changed(self, value):
        if value:
            g.config.frame = int(value)

    @values_changed
    def on_start_id_changed(self, value):
        if value:
            g.config.start_id = int(value)
            if g.config.start_id >= g.config.final_id:
                self.elements["results"].setText("Invalid start and final IDs.")
            else:
                self.elements["results"].setText(" ")
                g.tracked = False

    @values_changed
    def on_final_id_changed(self, value):
        if value:
            g.config.final_id = int(value)
            if g.config.final_id <= g.config.start_id:
                self.elements["results"].setText("Invalid start and final IDs.")
            else:
                self.elements["results"].setText(" ")
                g.tracked = False

    """
        Displaying images
    """
    def display_image(self, cv_im, bboxes = None, color=(255, 0, 255), frame_id=0):
        height = cv_im.shape[0]
        width = cv_im.shape[1]

        # calculating new width and height
        if width > height:
            newWidth = self.elements["image"].width() - self.layout().spacing()
            newHeight = int(height * (newWidth / width))
        else:
            newHeight = self.elements["image"].height() - self.layout().spacing()
            newWidth = int(width * (newHeight / height))

        # resize CV image
        cv_im = cv2.resize(cv_im, (newWidth, newHeight))

        # draw rectangles
        if bboxes is not None:
            widthMod = (newWidth / width)
            heightMod = (newHeight / height)

            for index, box in bboxes.iterrows():
                if box["xi"] > width or box["xj"] > width or box["yi"] > height or box["yj"] > height: continue # bugged objects
                else:
                    cv2.rectangle(cv_im, (round(box["xi"] * widthMod), round(box["yi"] * heightMod)), 
                        (round(box["xj"] * widthMod), round(box["yj"] * heightMod)), color=color)

        # display
        pixmap = convertCvImage2QtImage(cv_im)
        self.elements["image"].setPixmap(pixmap)
        self.elements["image"].repaint()

    def on_display_click(self):
        fig = g.parser.load_frames_id(g.config.frame, rect=False)
        bboxes = g.parser.bbox_data.loc[g.parser.bbox_data["frame_id"] == g.config.frame+1][["xi", "yi", "xj", "yj"]]
        self.display_image(fig, bboxes)
        g.tracked = False

    def clear_display(self):
        self.elements["image"].setText("No image loaded.")
        g.tracked = False

    """
        Updating frame and sequence options
    """
    def update_sequence(self):
        path = os.path.join(g.config.mot_path, g.config.folder)
        seq = sorted([x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))])
        self.elements["sequence"].clear()
        self.elements["sequence"].addItems(seq)

    def update_frame(self):
        path = os.path.join(g.config.mot_path, g.config.folder, g.config.sequence, 'img')
        frames = sorted([x.split(".")[0] for x in os.listdir(path)])
        self.elements["display_frame_id"].clear()
        self.elements["display_frame_id"].addItems(frames)

        frames = frames[1:-1]
        self.elements["start_frame_id"].clear()
        self.elements["start_frame_id"].addItems(frames) # slicing removes first and final frame
        self.elements["end_frame_id"].clear()
        self.elements["end_frame_id"].addItems(frames)

        self.elements["end_frame_id"].setCurrentText(frames[-1])

        g.tracked = False

    """
        Creates a popup to choose a dir
    """
    def choose_mot_path(self):
        path = QFileDialog.getExistingDirectory(self, 'Pick a valid path to the MOT folder.')
        if path: 
            g.config.mot_path = path
            self.update_elements()
        else: self.choose_mot_path()

    """
        Method to display text in the results section
    """
    def display_text(self, text: str):
        self.elements["results"].setText(text)
        self.elements["results"].repaint()

            