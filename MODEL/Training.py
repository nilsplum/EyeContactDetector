import logging

import torch
from torch import nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from MODEL.Data import EyeDataset
from MODEL.Model import EyeContactNetwork
import traceback

def train_model(num_epochs, batch_size, model_path=None):
    """
        Trains the EyeContactNetwork model using the specified parameters and dataset. Supports resuming training from a
        checkpoint if provided. Saves the model's state and optimizer's state at the end of each epoch.

        The function initializes the model, prepares the dataset and dataloader, sets up the loss function and optimizer,
        and iterates over the dataset for the specified number of epochs. Training loss is logged, and a plot of the loss
        across epochs is generated at the end.

        Parameters:
        - num_epochs (int): The number of epochs to train the model.
        - batch_size (int): The batch size used for training.
        - model_path (str, optional): The path to save the model checkpoint. If a checkpoint exists at this path, training
                                      will resume from there. If not provided, a default path is used.

        This function is crucial for model training, allowing for flexible training sessions and providing insights into
        the training process through loss visualization. It also ensures the model is saved periodically, preventing
        data loss during long training sessions.

        The dataset is loaded from a predefined folder, and the DataLoader shuffles it to ensure variability in the training
        batches. The Binary Cross-Entropy Loss (BCELoss) is used for training the binary classification (eye contact detection) model.
        The Adam optimizer is employed for optimization due to its efficiency in handling sparse gradients and adaptive learning rates.

        If a model_path is specified and a checkpoint is found, the training resumes from the last epoch saved in the checkpoint,
        maintaining the continuity of the training process. Otherwise, the training starts from scratch, and the model is saved
        to either the specified path or a default location.
    """
    model = EyeContactNetwork()
    data_folder = "./Data"
    dataset = EyeDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    already_trained_epochs = 0

    epochs = [i for i in range(0, num_epochs)]
    losses = []

    final_model_path = None
    if model_path is not None and os.path.isfile(model_path):
            final_model_path = model_path
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            already_trained_epochs = checkpoint['epoch']
            logging.info(f"Model has lastly been trained for {already_trained_epochs + 1} epochs")
    elif model_path is not None:
        final_model_path = model_path
        logging.info(f"Creating model file called '{model_path}'")
        fp = open(final_model_path, 'x')
        fp.close()
    model_failure = False

    try:
        for right_eye_images, left_eye_images, head_positions, new_labels in dataloader:
            pass
    except Exception as e:
        model_failure = True

    if model_path is None or model_failure:
        final_model_path = "MODEL/model.pt"
        logging.info("No checkpoint found. Starting from scratch by replacing default model.pt file")
        if os.path.isfile(final_model_path):
            logging.info("Removed model.pt file!")
            os.remove(final_model_path)

        logging.info("Created model.pt file!")
        fp = open(final_model_path, 'x')
        fp.close()

    for epoch in range(0, num_epochs):
        running_loss = 0.0
        try:
            for right_eye_images, left_eye_images, head_positions, new_labels in dataloader:
                left_eye_images.unsqueeze(1)  # Extrahieren des linken Auges
                right_eye_images.unsqueeze(1)  # Extrahieren des rechten Auges

                labels = [1 if label == "True" else 0 for label in new_labels]
                labels = torch.tensor(labels).unsqueeze(1).float()

                optimizer.zero_grad()

                outputs = model(left_eye_images, right_eye_images, head_positions)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = (running_loss / len(dataloader))*100
            losses.append(epoch_loss)
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            checkpoint = {
                'epoch': epoch + already_trained_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

            torch.save(checkpoint, final_model_path)
        except Exception as e:
            print("test")

    show_training_analysis(epochs, losses)

    logging.info("Training finished!")

def show_training_analysis(epochs, losses):
    """
        Generates and displays a plot showing the training loss over epochs. This visualization helps in understanding the
        training process and assessing the model's learning performance over time.

        Parameters:
        - epochs (list): A list of epoch indices.
        - losses (list): A list of loss values corresponding to each epoch.

        The function plots the epochs against the losses with specific styling options to enhance readability. The red
        markers indicate the loss value at each epoch, providing a clear visual cue of the training trend. This plot is
        instrumental for diagnosing training issues such as overfitting or underfitting and for making informed decisions
        about further training or hyperparameter adjustments.
    """
    plt.plot(epochs, losses, c="blue", marker='o', markersize=4, markerfacecolor='red')
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.title("TRAINING ANALYSIS")
    plt.show()




