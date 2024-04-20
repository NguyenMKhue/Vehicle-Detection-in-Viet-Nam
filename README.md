## Overview
This repository contains a system designed for vehicle and license plate detection, which includes data manipulation and machine learning model training. The project is structured to facilitate the development and evaluation of detection algorithms, making it suitable for both research and production use.

## Project Structure
The project is organized into specific folders to maintain a clean and efficient workflow:
- `workspace/`: Contains core scripts and utilities for the overall data processing pipeline.
  - `add_missing_data.py`: Script to add missing data points to datasets.
  - `util.py`: Utility functions for general purpose use across the project.
  - `visualize.py`: Visualization utilities for analyzing data and model outcomes.
- `license_plate_program/`: Dedicated to functionalities related to license plate detection.
  - `preprocess_license_plates.ipynb`: Notebook for preprocessing license plate images.
  - `license_plate_detector.pt`: Pre-trained model for detecting license plates.
  - `yolov8n.pt`: Pre-trained YOLOv8 model for object detection tasks.
  - `train.py`: Script to train the model on license plate detection.
  - `main.py`: Main script to run license plate detection tasks.
  - `config.yaml`: Configuration file for setting up model parameters and training environments.
- `vehicle_program/`: Scripts and notebooks for vehicle detection.
  - `Vehicle_Detector.ipynb`: Notebook that includes model training and detection for vehicles.
  - `preprocess_vehicle.ipynb`: Notebook for preprocessing vehicle data.
  - `train_val_test_splitting.ipynb`: Splits the vehicle data into training, validation, and test sets.

