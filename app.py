import copy
import logging
from datetime import datetime

import cv2

from EyeAndFaceTracking.EyeAndFaceTracker import FaceAndEyeTracker
import threading
import time

from MODEL import Evaluator
from MODEL.Training import train_model
from DataManaging.TrainingDataCollector import TrainingDataCollector

from PIL import Image


def run_eye_contact_tracker(model_path="MODEL/model.pt", print_prediction=True):
    """
        Initializes the eye and face tracker and continuously evaluates if the person looks into the camera using a pre-trained model.
        Predictions and their visual representation are buffered to reduce noise and improve detection reliability.

        Args:
        - model_path (str): Path to the trained model file for eye contact prediction (if the person looks into the camera)
        - print_prediction (bool): Flag indicating whether to print the prediction result to the console.

        The method captures frames at a high frequency, extracts eye pixels and facial data, and uses these for the prediction.
        A buffer (`last_10_buffer`) maintains the last 10 prediction results to stabilize detection by filtering out noise.

        Data taken into account:
            - Both eye pixel data
            - Head position in a 3D space -> "gaze direction" (Can be removed)
    """
    eye_and_face_tracker = FaceAndEyeTracker()
    eye_and_face_tracker.init_eye_tracker()

    # With this buffer the programm can detect if somebody really looks into the camera to reduce noise in the data
    last_10_buffer = []

    time.sleep(5)  # TODO replace with not None Bedingung für die Daten der Kamera
    logging.info("Tracking started")
    while True:
        # Es soll die bewegung angucken können, deswegen werden immer 5 bilder zusammengefasst
        time.sleep(0.0001)
        try:

            eye_left, eye_right = eye_and_face_tracker.get_current_eye_pixels()
            l_facial_data = eye_and_face_tracker.get_current_face_data()
        except:
            continue

        if (eye_left is not None) and (eye_right is not None) and (l_facial_data is not None):
            im_left = Image.fromarray(eye_left)
            im_right = Image.fromarray(eye_right)
            prediction, _ = Evaluator.predict_eye_contact(model_path, im_right, im_left, l_facial_data)
            if prediction:
                # Update the buffer
                if 0 < len(last_10_buffer) > 9:
                    last_10_buffer.pop(0)
                last_10_buffer.append(1)
                # Print a long bar for better visibility in a test scenario
                if print_prediction:
                    print("###############################################################################################")

            else:
                if len(last_10_buffer) > 9:
                    last_10_buffer.pop(0)
                last_10_buffer.append(0)
                if print_prediction:
                    print("######")
        else:
            if 0 < len(last_10_buffer) > 9:
                last_10_buffer.pop(0)
            last_10_buffer.append(0)
            if print_prediction:
                print("no face")

        active_odds = 0
        for active in last_10_buffer:  # Just checking if the list has more then 8 True
            active_odds += active

        if active_odds > 5:
            if print_prediction:
                print(f"{active_odds * 10}")


def show_opencv():
    """
      Initializes and displays real-time eye and face tracking data using OpenCV.
      This function is primarily used for visual debugging and demonstration purposes.
      It disables threading for the eye and face tracker to run synchronously within the main thread.
    """
    eye_and_face_tracker = FaceAndEyeTracker()
    eye_and_face_tracker.init_eye_tracker(False)


def init_logging():
    """
        Configures the logging system to report informational messages.
        This setup is essential for tracking the flow of the application and diagnosing issues.
    """
    logging.basicConfig(level=logging.INFO)


def train(num_epochs, batch_size, data_folder, model_path=None):
    """
        Initiates the training process for the prediction model.
        Args:
            - num_epochs (int): Number of training epochs.
            - batch_size (int): Size of batches used during training.
            - model_path (str, optional): Custom path to save the trained model. If not specified, a default path is used.
        This function abstracts the details of the training process, including data loading, model optimization, and saving.
    """
    if model_path:
        train_model(num_epochs, batch_size, data_folder, model_path)
    else:
        train_model(num_epochs, batch_size, data_folder)


def run(show_camera=True):
    """
        Starts the gaze tracking and prediction system in a separate thread and optionally shows the camera feed.
        Args:
            - show_camera (bool): If True, opens a window showing the camera feed with face and eye tracking overlays.
        This function is the main entry point to start gaze tracking and eye contact prediction functionalities.
    """
    threading.Thread(target=run_eye_contact_tracker).start()
    if show_camera:
        show_opencv()


def collect_data():
    """
        Starts the data collection process for training data acquisition.
        This function initiates the TrainingDataCollector, which handles capturing, processing, and saving facial data and eye images.
        Instructions for usage and control are displayed in the console, guiding the user through the data collection process.
    """
    # Note that if you try to show opencv, the marks get saved in the pixeldata
    collector = TrainingDataCollector("Data")
    collector.start()


""" 
    Have in mind that the training data is really limited and in order to get good results the following is necessary:
        - More diverse training data (can be generated via the collect data method):
            - Different lighting conditions
            - Different people (but only one in the image at the same time during data collectionch)
            -> Data should get reduced via DataShortener to reduce the amount in a clever way 
        - High quality image (at least 4K footage)
"""
# Example of using the pretrained model
init_logging()

# if you remove the path, the default model is overwritten
# Backups of the model and the data are in "BACKUP" directory
train(30, 15, "MODEL/model.pt")
