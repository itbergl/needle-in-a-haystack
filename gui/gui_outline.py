"""
    Stores the outline for the GUI via a new class and dictionaries.

    The `Widg` class stores positional info and adds a constructor that creates the QObject. If the 
    object is a button, adds the name to the text.

"""

from PySide6.QtWidgets import (QLabel, QPushButton, QComboBox, QLineEdit)
from .initialisers import *

class Widg():
    """
        Holds information about widgets - almost a dictionary
    """
    def __init__(self, name: str, Qtype, row_pos, col_pos, row_span, col_span, initialiser = None):
        self.name = name
        self.Qtype = Qtype
        self.pos = [row_pos, col_pos, row_span, col_span]
        self.init = initialiser

    def construct(self):
        """
            Returns the QWidget object after construction
            IF self.qtype is QPushButton, add the name to the constructor
        """
        if self.Qtype is QPushButton: return self.Qtype(self.name)
        else: return self.Qtype()

"""
    OUTLINE of the GUI
    Dictionary of categories, with several objs per category
    Each object key should be unique
"""
OUTLINE = {
    "config": {
        "config_title": Widg("Configuration", QLabel, 0, 0, 1, 2),

        "mot_path_name": Widg("  MOT Path", QLabel, 1, 0, 1, 1),
        "mot_path": Widg("Choose MOT Path", QPushButton, 1, 1, 1, 1),

        "folder_name": Widg("  Folder Name", QLabel, 2, 0, 1, 1),
        "folder": Widg("folder", QComboBox, 2, 1, 1, 1),

        "sequence_name":Widg("  Sequence Num", QLabel, 3, 0, 1, 1),
        "sequence": Widg("sequence", QComboBox, 3, 1, 1, 1),

        "max_ram_name": Widg("  Max RAM (GB)", QLabel, 4, 0, 1, 1),
        "max_ram": Widg("max_ram", QLineEdit, 4, 1, 1, 1, max_ram)
    },
    "solutions": {
        "solutions_title": Widg("Solutions", QLabel, 5, 0, 1, 2),

        "display_frame_id_name": Widg("  Frame ID", QLabel, 6, 0, 1, 1),
        "display_frame_id": Widg("display_frame_id", QComboBox, 6, 1, 1, 1),
    
        "display": Widg("Display", QPushButton, 7, 0, 1, 2),
        "clear_display": Widg("Clear Display", QPushButton, 8, 0, 1, 2)
    },
    "hyperparams": {
        "hyperparam_title": Widg("Hyperparameters", QLabel, 0, 2, 1, 2),

        "tau_name": Widg("Tau", QLabel, 1, 2, 1, 1),
        "tau": Widg("tau", QLineEdit, 1, 3, 1, 1, tau),

        "sigma_p_name": Widg("Sigma P", QLabel, 2, 2, 1, 1),
        "sigma_p": Widg("sigma_p", QLineEdit, 2, 3, 1, 1),

        "sigma_v_name": Widg("Sigma V", QLabel, 3, 2, 1, 1),
        "sigma_v": Widg("sigma_v", QLineEdit, 3, 3, 1, 1),

        "sigma_a_name": Widg("Sigma A", QLabel, 4, 2, 1, 1),
        "sigma_a": Widg("sigma_a", QLineEdit, 4, 3, 1, 1),

        "sigma_d_name": Widg("Sigma D", QLabel, 5, 2, 1, 1),
        "sigma_d": Widg("sigma_d", QLineEdit, 5, 3, 1, 1, sigma)
    },
    "tracking": {
        "tracking_title": Widg("Tracking", QLabel, 9, 0, 1, 4),

        "start_frame_id_name": Widg("  First Frame ID", QLabel, 10, 0, 1, 1),
        "start_frame_id": Widg("start_frame_id", QComboBox, 10, 1, 1, 1),

        "end_frame_id_name": Widg("  Final Frame ID", QLabel, 11, 0, 1, 1),
        "end_frame_id": Widg("end_frame_id", QComboBox, 11, 1, 1, 1),

        "track": Widg("Track and\nEvaluate", QPushButton, 10, 2, 2, 2)
    },
    "results": {
        "results_title": Widg("Results", QLabel, 12, 0, 1, 4),
        "results": Widg(" ", QLabel, 13, 0, 4, 4),
        "save_results": Widg("Save Results", QPushButton, 17, 0, 1, 2, save_results),
        "save_gif": Widg("Save GIF", QPushButton, 17, 2, 1, 2, save_gif)
    },
    "image": {
        "image_title": Widg("Image", QLabel, 0, 4, 1, 7),
        "image": Widg("Image", QLabel, 1, 4, 17, 7)
    }
}