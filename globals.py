"""
    Global parameters
"""

class Config():
    """
        Configuration variables
    """
    folder = None
    mot_path = None
    sequence = None

    frame = None
    start_id = None
    final_id = None 

    def __init__(self):
        self.max_ram = 2 # ram in GB

class Obj():
    """
        Stores variables relating to candidate detection and discrimination
    """
    det = None
    disc = None
    masks = None
    candidate_masks = None

class Hyperparameters():
    """
        Stores hyperparameters that are set to defaults
    """
    def __init__(self):
        self.tau = 1
        self.sigmas = [1,1,1,1] # p, v, a, d

class G():
    def __init__(self):
        # storage objects; storing particular variables
        self.config = Config()
        self.obj = Obj()
        self.hyperparams = Hyperparameters()

        # objects required globally
        self.parser = None
        self.gui = None
        self.m = None # matrics object

        # tracking
        self.tracker = None
        self.tracked = False
        self.tracking_results = None

# initialise g
g = G()