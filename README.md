# Driver Drowsiness Detection System

This project implements a driver drowsiness detection system using computer vision and deep learning. The system can detect when a driver is becoming drowsy by monitoring their eye state and facial patterns.

## Features

- Real-time drowsiness detection using a webcam
- Two detection methods:
  - Eye Aspect Ratio (EAR) calculation
  - CNN-based eye state classification
- Tools for collecting and preparing custom training data
- Training script for the eye state classifier

## Requirements

Install the required dependencies:

```
pip install -r requirements.txt
```

Additionally, you need to download the facial landmark predictor from dlib:
1. Download the shape predictor from [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. Extract the file and place it in the `models` directory

## Project Structure

```
├── data/                  # Data directory
│   ├── train/             # Training data
│   │   ├── open/          # Open eye images
│   │   └── closed/        # Closed eye images  
│   └── validation/        # Validation data
│       ├── open/          # Open eye images
│       └── closed/        # Closed eye images
├── models/                # Trained models
├── src/                   # Source code
│   ├── drowsiness_model.py       # Model architecture
│   ├── drowsiness_detection.py   # Main detection script
│   ├── prepare_data.py           # Data collection script
│   └── train_model.py            # Model training script
└── requirements.txt       # Dependencies
```

## Usage

### 1. Prepare Dataset (Optional)

If you want to train your own model, you need to collect eye images:

```
python src/prepare_data.py
```

This will open your webcam and guide you through the data collection process.
- Press 's' to start/stop collecting data
- Press 'o' to switch to collecting 'open' eye images
- Press 'c' to switch to collecting 'closed' eye images
- Press 'q' to quit

### 2. Train Model (Optional)

If you've collected your data or have your own dataset:

```
python src/train_model.py
```

This will train a CNN model to classify between open and closed eyes.

### 3. Run Drowsiness Detection

Run the main drowsiness detection script:

```
python src/drowsiness_detection.py
```

By default, the system uses the Eye Aspect Ratio method. To use the CNN model:

```
python src/drowsiness_detection.py --use_ear False
```

## How It Works

### Method 1: Eye Aspect Ratio (EAR)

EAR measures the ratio of the distances between the vertical and horizontal eye landmarks. When someone is drowsy, their eyes start to close, decreasing the EAR value. If the EAR falls below a threshold for a certain number of consecutive frames, the system triggers a drowsiness alert.

### Method 2: CNN Classification

The CNN model is trained to directly classify eye images as 'open' or 'closed'. The model analyzes eye regions extracted from the webcam feed and predicts the state. If both eyes are detected as closed for several consecutive frames, the system triggers an alert.

## Customization

- Adjust the EAR threshold in `drowsiness_model.py` to make the system more or less sensitive
- Modify the consecutive frames threshold to change how long eyes need to be closed before an alert
- Train the CNN model on your own data for better accuracy

## License

This project is open source and available under the MIT License. 