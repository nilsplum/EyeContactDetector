import logging
import os
import threading
import time
from pynput import keyboard
from PIL import Image
from EyeAndFaceTracking.EyeAndFaceTracker import FaceAndEyeTracker


class TrainingDataCollector:
    def __init__(self, data_dir, delay=0.1):
        self.data_dir = data_dir
        self.is_eye_contact = False
        self.delay = delay

    def datacollector(self):
        """
        Collects data for eye contact detection by periodically capturing eye images and corresponding head position data.

        This function sets up an EyeAndFaceTracker instance to continuously capture images of the left and right eyes along
        with head pose data. It saves this data to disk, labeling each image according to whether eye contact is detected based
        on user input (spacebar press).

        Parameters:
        - delay (float): The delay in seconds between each data capture cycle.

        The function listens for a spacebar press to toggle the eye contact state and saves pairs of eye images with metadata
        indicating the head position and whether eye contact was made. Images are saved in a directory named 'Data', and each
        pair is uniquely numbered with an incrementing identifier. The process runs indefinitely until manually stopped.

        Users are prompted to indicate when they are looking into the camera by pressing the spacebar, facilitating the
        collection of labeled data for training a model to detect eye contact.
        """
        eye_and_face_tracker = FaceAndEyeTracker()
        eye_and_face_tracker.init_eye_tracker()

        logging.info("PRESS SPACEBAR IF YOU ARE LOOKING INTO THE CAMERA")
        time.sleep(1)
        logging.info("Starting in:")
        logging.info("5")
        time.sleep(1)
        logging.info("4")
        time.sleep(1)
        logging.info("3")
        time.sleep(1)
        logging.info("2")
        time.sleep(1)
        logging.info("1")
        logging.info("Starting to collect the data")

        image_count = 0
        while True:
            time.sleep(self.delay)
            all_files = os.listdir(self.data_dir)
            if ".DS_Store" in all_files:
                all_files.remove(".DS_Store")
            max_file_number = 0
            for filename in all_files:
                file_number = int(filename.split("_")[0])
                if file_number > max_file_number:
                    max_file_number = file_number

            try:
                temp_is_eyecontact = self.is_eye_contact
                eye_left, eye_right = eye_and_face_tracker.get_current_eye_pixels()
                l_facial_data = eye_and_face_tracker.get_current_face_data()
                if (eye_left is not None) and (eye_right is not None) and (l_facial_data is not None):
                    im_left = Image.fromarray(eye_left)
                    im_right = Image.fromarray(eye_right)
                    image_left_name = f"{max_file_number + 1}_{temp_is_eyecontact}_left_{l_facial_data[0]}_{l_facial_data[1]}_{l_facial_data[2]}"
                    image_right_name = f"{max_file_number + 1}_{temp_is_eyecontact}_right_{l_facial_data[0]}_{l_facial_data[1]}_{l_facial_data[2]}"

                    im_left.save(f"{self.data_dir}/{image_left_name}.png")
                    im_right.save(f"{self.data_dir}/{image_right_name}.png")
                else:
                    logging.info("No face detected.")

                logging.info(f"Collected image pair: {image_count}")
                image_count += 1
            except Exception as e:
                logging.error(f"Error during data collection: {e}")

    def start(self):
        """
        Starts the data collection process in a separate thread and sets up keyboard listeners to detect eye contact events.

        Parameters:
        - delay (float): The time interval in seconds between data captures.

        This function initiates a parallel thread that runs the datacollector function for continuous data capture. It also
        listens for keyboard events to toggle the eye contact detection flag (`is_eyecontact`) based on specific key presses
        (spacebar to indicate eye contact, esc to stop the listener). This setup allows for real-time labeling of the data
        based on user interaction, enabling the collection of accurately labeled data for training eye contact detection models.
        """
        threading.Thread(target=self.datacollector).start()

        def on_press(key):
            if key == keyboard.Key.space:
                self.is_eye_contact = not self.is_eye_contact
            elif key == keyboard.Key.esc:
                # Stop listener
                return False

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = TrainingDataCollector("../Data")
    collector.start()
