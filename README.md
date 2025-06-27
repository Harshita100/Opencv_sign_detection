# Hand Gesture Recognition using MediaPipe and Random Forest

This project detects and recognizes hand gestures in real-time using MediaPipe for hand landmark detection and a Random Forest classifier for gesture prediction.

## Features

- Real-time hand tracking using webcam
- Landmark extraction using MediaPipe Hands
- Gesture normalization for consistent recognition
- Gesture classification using Random Forest
- Model training, saving, and loading with pickle
- Visual feedback with bounding box and gesture label on the video feed


## Dataset Collection

- Images are collected per gesture class and stored in folders under
    `./data/<gesture_name>/`
- Each image is processed to extract 21 hand landmarks (x, y coordinates).
- Coordinates are normalized relative to the hand’s bounding box for position-independent recognition.
- Saved as data.pickle with structure:
    >`{'data': [...], 'labels': [...]}`

## Model Training

- Model training uses a RandomForestClassifier from sklearn.
- The data is split into training and testing sets.
- After training, the model is saved using pickle as data_tested.pickle.

## Real-Time Gesture Prediction

- The trained model is loaded from data.pickle.
- The system captures frames from a webcam and detects hand landmarks using MediaPipe.
- It normalizes and predicts the gesture using the trained model.
- The bounding box and predicted gesture label are drawn on the live video feed.

## Dependencies

- This project uses the following libraries:
- opencv-python
- mediapipe
- numpy
- scikit-learn
- matplotlib
- pickle

> *Install them using pip if needed.*

## Run

- Collect training images per gesture in respective folders.
- Run the training script to generate the model.
- Start the real-time detection script to recognize gestures live.

## Example Gestures

- peace
- fist
- thumbs up

You can extend this project by adding more gesture classes and retraining the model.
