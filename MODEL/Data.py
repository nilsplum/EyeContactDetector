import logging

import PIL
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Definition der Transformationspipeline für die Bildvorverarbeitung
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Ändern Sie die Größe der Bilder auf 64x64 Pixel
    transforms.ToTensor(),  # Wandeln Sie die Bilder in Tensoren um
    transforms.Normalize((0.5,), (0.5,))  # Normalisieren Sie die Bildpixel auf den Bereich [-1, 1]
])


# Definition des Dataset-Klasse für den Ordner mit den Bildern
class EyeDataset(Dataset):
    """
        A custom dataset class for loading eye image pairs and their associated head positions from a specified folder.

        This dataset is designed to support training and evaluation of eye contact detection models by providing pairs of eye images
        (left and right) along with the head position at the time the image was taken. The dataset automatically handles
        the preprocessing of images, including resizing, conversion to tensor, and normalization.

        Attributes:
        - folder_path (str): Path to the folder containing the eye images.

        The dataset filters out non-eye images (e.g., .DS_Store files on macOS) and expects a specific naming convention
        for the images to associate left and right eyes with their corresponding head position and eye contact label.

        The naming convention for images should include a unique identifier for the image pair, followed by 'left' or 'right'
        to indicate the eye, and additional metadata for head position and "eye contact" with the camera. This metadata is used to generate
        labels and head position tensors for model training.

        Parameters:
        - folder_path (str): The file path to the directory containing the dataset images.

        Methods:
        - __len__: Returns the number of unique eye image pairs in the dataset.
        - __getitem__: Retrieves a specific item from the dataset by index, returning the preprocessed left and right eye images,
          the head position tensor, and the eye contact label ('True' for eye contact, 'False' otherwise).
    """

    def __init__(self, folder_path):
        """
            Initializes the dataset with the path to the folder containing the eye images, filtering out irrelevant files
            and preparing the list of unique identifiers for image pairs.
        """
        self.folder_path = folder_path
        self.image_file_names = os.listdir(folder_path)
        for image_file in self.image_file_names:
            if "DS_Store" in image_file:
                self.image_file_names.remove(image_file)
        # Extrahieren der IDs der zusammengehörigen Bildpaare
        self.ids = list(set([image_file.split("_")[0] for image_file in self.image_file_names]))

    def __len__(self):
        """
            Calculates the total number of unique eye image pairs available in the dataset.
            Returns:
            - int: The number of unique eye image pairs.
        """
        return len(self.ids)

    def __getitem__(self, index):
        """
            Retrieves a specific item from the dataset by its index, providing preprocessed eye images, head position,
            and eye contact label.

            This method is responsible for loading the images, performing preprocessing steps, and extracting metadata
            for head position and eye contact (True/False). In case of file-related errors, it performs cleanup operations and logs errors.

            Parameter index: An integer index corresponding to the eye image pair to be retrieved from the dataset.

            returns: A tuple containing the following elements:
                      - left_eye_image: A tensor representing the preprocessed left eye image.
                      - right_eye_image: A tensor representing the preprocessed right eye image.
                      - head_position: A tensor of floats representing the head position at the time of the image capture.
                      - label: A string ('True' or 'False') indicating whether the person looks into the camera (True) or not (False).

            Note: In case of missing or corrupted image files for a specific index, the method attempts to clean up by deleting
            problematic files and recursively calls itself with an adjusted index to retrieve the next valid image pair.
        """
        try:
            id = self.ids[index]
            left_eye_image_path, right_eye_image_path = None, None

            for temp_name in self.image_file_names:
                if temp_name.split("_")[0] == id:
                    if "left" in temp_name:
                        left_eye_image_path = os.path.join(self.folder_path, temp_name)
                    else:
                        right_eye_image_path = os.path.join(self.folder_path, temp_name)
            try:
                left_eye_image = Image.open(left_eye_image_path).convert("L")
                right_eye_image = Image.open(right_eye_image_path).convert("L")

                left_eye_image = transform(left_eye_image)
                right_eye_image = transform(right_eye_image)

                left_eye_image_path_split = left_eye_image_path.split("_")
                head_position = [float(left_eye_image_path_split[3]), float(left_eye_image_path_split[4]),
                                 float(left_eye_image_path_split[5].replace(".png", ""))]
                head_position = torch.tensor(head_position, dtype=torch.float32)

                label = "True" if "True" in left_eye_image_path else "False"
            except Exception as e:
                if isinstance(e, PIL.UnidentifiedImageError):
                    logging.error(f"IMAGE PAIR WITH ID: {id} NOT AVAILABLE!")
                    try:
                        if left_eye_image_path is not None:
                            os.remove(left_eye_image_path)
                            logging.info(f"Data automatically cleaned: {str(left_eye_image_path)} DELETED!")
                        if right_eye_image_path is not None:
                            os.remove(right_eye_image_path)
                            logging.info(f"Data automatically cleaned: {str(right_eye_image_path)} DELETED!")
                    except FileNotFoundError as e:
                        logging.error(e)
                    # Just take another image
                    index = index - 1
                    left_eye_image, right_eye_image, head_position, label = self.__getitem__(index)
                    return left_eye_image, right_eye_image, head_position, label
                else:
                    logging.error(f"IMAGE ERROR: {e}")

        except Exception as e:
            print("e")

