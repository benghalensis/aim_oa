# Object Tracking and Grid-Based Simulation

This repository contains two main components:

1. **Object Tracking**: A particle filter-based object tracking system for detecting and tracking objects (e.g., balls) in video sequences using Detectron2 and custom tracking logic.
2. **Grid-Based Simulation**: Answer for Task 1: Thief and cops

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:benghalensis/aim_oa.git
   cd aim_oa
   ```

2. Create a Conda environment and install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate vision
   ```

3. Install Detectron2:
   Follow the [Detectron2 installation guide](https://detectron2.readthedocs.io/tutorials/install.html) to set up Detectron2 in your environment.

## Usage

### Object Tracking (`run_task_2.py`)

This script tracks a ball in a video sequence using Detectron2 and a particle filter.

1. Prepare the input frames by extracting them from a video file. Place the frames in the `data/frames/` directory:
   ```bash
   mkdir -p data/frames
   ffmpeg -i ball_tracking_video.mp4 data/frames/frame_%04d.png
   ```
2. Run the script:
   ```bash
   python run_task_2.py
   ```
3. Results:
   - Tracked frames are saved in `data/results/`.
   - Tracking data is saved in `results.csv`.
   - Debug visualizations are saved in `data/debug/`.

### Grid-Based Simulation (`run_task_1.py`)

This script simulates a grid-based environment where cops attempt to locate a thief.

1. Run the script:
   ```bash
   python run_task_1.py
   ```
2. The script includes several test cases to validate the simulation logic.

## Results

The results of the object tracking can be found [here](https://drive.google.com/drive/folders/1wMKb17ELVAmKOc8Nw05ccxRbGigs-aVn?usp=drive_link).

## Key Features

- **Object Tracking**:
  - Detectron2-based object detection.
  - Particle filter for robust tracking.
  - Debugging visualizations for tracking states.

- **Grid-Based Simulation**:
  - Simulates visibility and movement in a grid.
  - Handles multiple cops with varying orientations and fields of view.
  - Identifies the nearest cell to the thief that is not visible to any cop.

## Dependencies

- Python 3.9
- Detectron2
- NumPy
- OpenCV
- PyTorch
- Scikit-learn
- YAML

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Acknowledgments

- [Detectron2](https://github.com/facebookresearch/detectron2) for object detection.
- Contributors to this repository for their efforts in developing the tracking and simulation systems.