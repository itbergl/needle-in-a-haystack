# CITS4402 Project: Object Tracking from Satelite Images

**Authors:**

-   Dhruv Jobanputra (22704304)
-   Aditya Gupta (22723081)
-   Harrison Mueller (22732927)

## Description

The aim of this project was to implement the tracking algorithm presented by Wei _et al._ in **Needles in a Haystack: Tracking City-Scale Moving Vehicles From Continuously Moving Satellite**. The algorithm was implemented successfully, and a GUI was implemented that allows an end-user to track moving objects, and evaluate the tracking based on the VISO dataset.

Unfortunately, due to the complexity of this project, and time constraints, we were not able to implement everything efficiently and accurately. The calibration of candidate discrimination has not been implemented, and the overall tracking is fairly inaccurate. In terms of efficiency, ideally we would use numpy functions more heavily, particularly for looping through frames, candidates, etc; however, we have not had time to improve this.

Additionally, there is a known bug when running on MacOSx where the UI does not update for tracking and evaluation. On other operating systems, such as Linux, a dialogue is presented in the `Results` section that tells the user what process is being run, and after evaluation has completed, it shows the statistics. For each frame, an image is also shown with tracking boxes drawn.

## Running

### Creating a Python Virtual Environment

Ensure you have the [python virtual environment](https://docs.python.org/3/library/venv.html) installed, along with [pip3](https://pypi.org/project/pip/).

#### Steps

1.  Create the environment

```
python3 -m venv venv
```

2.  Load environment

```
source venv/bin/activate 	 # linux / mac
venv\Scripts\activate.bat 	# windows
```

3.  Install the required packages

```
pip install -r requirements.txt
```

### Running `main.py`

Call:
`python3 main.py`

`main.py` can also be called with a path to a `mot` directory (`python3 main.py <dir>`). If not provided, the default is `mot`. If this is not a valid directory, a pop-up will request a valid directory.

## Usage Instructions

Upon running the script, the user will be presented with a GUI that has several available options, each divided into several sections.

### Configuration

The options in this section are used to set up the tracking.

-   `MOT Path`: Allows the user to change the file system path to the `mot` folder of the VISO dataset
-   `Folder Name`: Specifies the name of the subfolder of `mot`
-   `Sequence Num`: Specify the sequence number
-   `Max RAM (GB)`: Specifies the maximum amount of RAM that can be used to store images. This is used for tracking multiple files; if all images within a track can be loaded into the main memory, they will be. Beware, there are other processes that are not affected by this limit.

### Hyperparameters

These options allow the user to tune the hyperparameters for tracking, as described by Wei _et al._.

### Solutions

These options allow a user to view the solution to a given image. Upon selecting a frame ID, the user can press the `Display` button to show the image with tracking boxes, as defined in `gt.txt`, drawn in green.

### Tracking

A user can select a first frame ID and a final ID, both of which are inclusive, that they wish to apply the tracking algorithm to. The first and final frames of the dataset are removed as tracking cannot be applied to these images. Upon selection of frame IDs, a user can press the `Track and Evaluate` button, which starts the tracking algorithm. The final frame must be after the first frame.

Progress messages are displayed in the `Results` section, and as the tracking is evaluated, they will appear in the `Image` section with tracking boxes drawn in green. Upon completion of tracking and evaluation, statistics will be shown in the results section.

### Results

Progress messages and statistics are shown here. If the user wishes to save the results to a file, they can select the `Save Results` button, which will show a popup. The user can then specify a filename that the CSV should be saved to.

### Image

The size of the image is defined by the size of the GUI. If the user wishes to decrease the GUI, the user needs to press the `Clear Display` button to remove the image prior to resizing.

## Design

This project was written in Python3. Despite heavy use of libraries such as Numpy and OpenCV2, the algorithm takes a while to run. Further improvements are possible, but due to time constraints, were not able to be implemented.

### File Structure

```
.
├── main
|   ├── gui
|   |   ├── gui_outline.py      Stores the positional info for the GUI
|   |   ├── initialisers.py     Contains some initialising functions for widgets on the GUI
|   |   └── gui.py              Constructs the GUI and handles user interactions
|   ├── object_detection
|   |   ├── detection.py        Stores the class that forms the candidate detection
|   |   └── discrimination.py   Stores the class that forms the candidate discrimination
|   ├── tracking
|   |   ├── evaluate.py         Evaluates the tracks found during tracking
|   |   ├── matrices.py         Class that stores the various matrices required for tracking
|   |   ├── object_tracker.py   Performs the tracking
|   |   ├── state_vector.py     Class the stores the state vectors and the covariance matrices, and provides update functions
|   |   └── track.py            Main execution file for tracking, calls candidate detection and discrimination, as well as evaluation
|   ├── globals.py          Provides global variables and classes that can be imported into any file
|   └── parser.py           Provides the class for parser image information
├── main.py             Main execution file; creates some objects and initialises the GUI
├── README.md           This file
└── requirements.txt    Package requirements
```

All `__init__.py` files are empty, but are required to allow subdirectories.

### Demo

![til](./demo.gif)
