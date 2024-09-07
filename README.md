 # Deep Face Detection with TensorFlow

This project implements a deep learning model for face detection using TensorFlow. The model is based on a modified version of the VGG16 architecture and performs both face classification (presence of a face) and face localization (bounding box prediction). 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
  
## Introduction
The goal of this project is to detect faces in images or video streams by training a deep neural network that simultaneously classifies the presence of a face and predicts bounding boxes around it.

Key components include:
- Custom data augmentation using the Albumentations library.
- Preprocessing pipelines using TensorFlow's `tf.data.Dataset`.
- A combined classification and regression model built on top of a pre-trained VGG16.

## Features
- **Dual Task**: Face detection and bounding box localization.
- **Data Augmentation**: Randomized augmentations applied to training data using Albumentations.
- **Customizable**: Flexible architecture for experimenting with different backbone networks.
- **End-to-End Pipeline**: From data loading to training and evaluation.

## Installation
### Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV 4.x
- Albumentations
- Numpy

### Steps
   Clone the repository:
   ```bash
   git clone https://github.com/Pragateeshwaran/Deep-Face-Detection.git
   cd deep-face-detection
   ```
 
## Usage
1. **Prepare your dataset** (see the [Dataset Preparation](#dataset-preparation) section).
2. **Run the training script**:
   ```bash
   python train.py
   ```

3. **Evaluate the model**:
   Use the provided script to test the trained model on new images or video streams.

## Dataset Preparation
- The dataset should consist of images and corresponding JSON files that include the bounding box coordinates for faces.
- Use the `cv2.VideoCapture` function (commented out in the script) to collect data from your webcam or other video sources.
- Bounding boxes should be provided in `[x_min, y_min, width, height]` format, relative to the image dimensions (640x480 in this case).

### Example Directory Structure:
```
├── data/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── labels/
│   │   ├── img1.json
│   │   ├── img2.json
```

## Model Architecture
- **Backbone**: The VGG16 architecture is used for feature extraction.
- **Classification Head**: A binary classification layer to predict whether a face is present.
- **Regression Head**: A regression layer to predict the bounding box coordinates.

The model outputs both the class score and the bounding box predictions.

## Training
- The model is trained using a custom training loop.
- The loss function consists of:
  - **Binary Cross-Entropy**: For the face classification task.
  - **Mean Squared Error (MSE)**: For the bounding box localization task.

### Augmentation
The Albumentations library is used to apply the following augmentations:
- Random flipping
- Random cropping
- Color adjustments

## Results
Once trained, the model can be used to detect faces in both images and videos. The bounding boxes around detected faces will be drawn on the output frames.
 
