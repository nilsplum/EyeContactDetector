import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from MODEL.Model import EyeContactNetwork


def predict_eye_contact(checkpoint_path, left_eye_image, right_eye_image, head_position):
    """
        Predicts whether eye contact is being made based on the provided left and right eye images and head position data.

        This function loads a pre-trained EyeContactNetwork model from the specified checkpoint, preprocesses the input images
        and head position data, and performs a forward pass through the model to predict eye contact.

        Parameters:
        - checkpoint_path (str): Path to the checkpoint file containing the saved model state.
        - left_eye_image (PIL.Image.Image): A PIL image instance of the left eye.
        - right_eye_image (PIL.Image.Image): A PIL image instance of the right eye.
        - head_position (list or tuple): A 3-element list or tuple containing the x, y, z coordinates of the head position.

        Returns:
        - tuple: A tuple containing a boolean indicating the presence of eye contact and the probability (as a float) associated with the prediction.

        The method converts the input PIL images to grayscale, applies a series of transformations including resizing, tensor
        conversion, and normalization to make them compatible with the model's input requirements. It then loads the model
        weights from the checkpoint, performs prediction, and interprets the output as either indicating eye contact or not
        based on a threshold. The function is robust to input validation, ensuring that all necessary inputs are provided before
        attempting to make a prediction.

        This approach allows for batch processing of eye images and head position data to determine if the person looks into the camera
        with high accuracy, leveraging deep learning for nuanced feature extraction and inference.
    """
    if checkpoint_path and left_eye_image and right_eye_image and head_position and head_position[0] and head_position[1] and head_position[2]:

        # Convert data to images
        left_eye_image = left_eye_image.convert("L")
        right_eye_image = right_eye_image.convert("L")

        head_position = torch.tensor(head_position, dtype=torch.float32)

        # Apply transformation pipeline
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        left_eye_image = transform(left_eye_image)
        right_eye_image = transform(right_eye_image)

        # Add additional batch dimension as wrapper in order to be able
        # to pass it to the forward method of the network
        left_eye_image = left_eye_image.unsqueeze(0)
        right_eye_image = right_eye_image.unsqueeze(0)
        head_position = head_position.unsqueeze(0)

        # Load model from checkpoint
        model = EyeContactNetwork()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get prediction
        test_output = model(left_eye_image, right_eye_image, head_position)
        test_prediction = test_output.item() > 0.5
        test_probability = test_output.item()

        return test_prediction, test_probability
    else:
        return False, 1




