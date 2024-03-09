import torch
from torch import nn


class EyeContactNetwork(nn.Module):
    """
        A neural network designed for determining eye contact by analyzing images of the left and right eyes alongside head position data.

        This network employs a sequence of convolutional layers to process the left and right eye images separately, extracting
        relevant features. These features are then concatenated with head position data and passed through fully connected
        layers to predict whether eye contact is being made.

        The architecture is structured to handle the spatial information in eye images efficiently while incorporating head
        position as an additional context for improving the accuracy of eye contact detection.

        The network uses ReLU activations for non-linearity and Sigmoid at the output to provide a probability indicating
        the likelihood of eye contact.

        Attributes:
        - conv_layers (nn.Sequential): Convolutional layers for feature extraction from eye images.
        - fc_layers (nn.Sequential): Fully connected layers for making the final prediction based on the concatenated features
          of both eye images and head position data.

        The forward pass requires separate tensors for the left eye, right eye, and head position, which allows the network
        to learn from both the appearance of the eyes and their relative positions to determine gaze direction and infer eye contact.
    """
    def __init__(self):
        """
            Initializes the EyeContactNetwork model by setting up the convolutional and fully connected layers.
        """
        super(EyeContactNetwork, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear((32 * 16 * 16 * 2)+3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, left_eye, right_eye, head_position):
        """
            Defines the forward pass of the model.

            Parameters:
            - left_eye (torch.Tensor): The tensor containing batched image data for the left eye. The tensor is expected to
                                       have a shape corresponding to (batch_size, channels, height, width).
            - right_eye (torch.Tensor): The tensor containing batched image data for the right eye, with the same shape
                                        expectations as the left_eye tensor.
            - head_position (torch.Tensor): A tensor containing batched head position data. The tensor shape should be
                                            (batch_size, 3) representing three-dimensional head position.

            Returns:
            - torch.Tensor: A tensor of shape (batch_size, 1) containing the predicted probability of eye contact for each
                            example in the batch.

            The method extracts features from both eye images using shared convolutional layers, flattens these features,
            and concatenates them with head position data. This combined feature vector is then passed through fully connected
            layers to predict eye contact probability.
        """
        left_eye_features = self.conv_layers(left_eye)
        left_eye_features = left_eye_features.view(left_eye_features.size(0), -1)

        right_eye_features = self.conv_layers(right_eye)
        right_eye_features = right_eye_features.view(right_eye_features.size(0), -1)

        combined_features = torch.cat((left_eye_features, right_eye_features, head_position), dim=1)
        output = self.fc_layers(combined_features)

        return output
