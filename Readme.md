# Real-Time Eye Contact Detection

## General Information

This project is designed to detect whether a person is looking directly into the camera, enabling real-time image classification of the camera feed into two classes: **True** (eye contact) and **False** (no eye contact). It encompasses a comprehensive workflow including data collection, data cleaning, model training, testing, and deployment for real-time applications.

Utilizing **OpenCV** for image processing and **PyTorch** for deep learning, the system analyzes pixel data around the eyes. This is achieved by extracting eye regions through OpenCV's landmark detection, followed by further processing and classification using a convolutional neural network (CNN). The model's architecture includes both convolutional and linear layers, with its performance open to enhancement through parameter tuning, activation function adjustments, and architectural modifications.

Note that this project serves as the basis of another project which I made working independently. This means the project is more like a template (only 2000 lines) and adaptable for other envirenments. For really good predictions it needs a bigger and more diverse dataset which is not included in this public version but can easily be created using the included `TrainingDataCollector` (example in the `app.py` file)

## How to Use

Keep in mind that the model template is trained on data of a specific image resolution, which means that inequality in resolution could lead to a greater loss. It is recommended to use a higher resolution camera and, of course, to make a larger and more diverse dataset. You should also use a different directory with test data to ensure the loss is calculated accurately.

The `app.py` file serves as the conceptual interface for interacting with the system, containing code snippets for various operations:

- **Running the Model**: Detailed instructions are provided for deploying the trained model for real-time eye contact detection.
- **Collecting Data**: The `TrainingDataCollector` allows for the automated collection of labeled eye images and head position data. The collection frequency can be adjusted using the `delay` parameter to suit different environments.
- **Training the Model**: The train method uses the `Training.py` file to train the model on the data inside the `Data Directory`.

A backup of the current model and data is in the `BACKUP` directory

## Use Cases

This project was intended to be used for the following purposes:
- Keywordless attention detection for virtual assistants
- Gesture tracking in combination with attention detection

## How It Works

The Real-Time Eye Contact Detection project is built upon a series of interconnected components that work together to capture, process, analyze, and classify images from a live camera feed. Below is an overview of how these components function together:

#### Machine Learning Model

- The core of the system is a **convolutional neural network (CNN)**, the `EyeContactNetwork`, designed to analyze the preprocessed eye images and head pose data to predict eye contact. The network comprises:
  - **Convolutional layers**: Extract features from the eye images. Each eye is processed separately, and the resulting features are concatenated.
  - **Fully connected layers**: Combine the features from both eyes with the head pose data to make a final prediction. The output is passed through a sigmoid activation function to obtain a probability indicating the likelihood of eye contact.
- The model can be trained using collected and preprocessed data. Training involves minimizing a binary cross-entropy loss function using an Adam optimizer.

#### Training and Prediction

- For **training**, the system uses a custom dataset class, `EyeDataset`, to load eye images and corresponding head pose data from a directory. The dataset is used in conjunction with a DataLoader for batch processing.
- **Model checkpoints** can be saved and loaded, allowing training to be paused and resumed as needed. This functionality supports incremental learning and model refinement.
- To **predict** eye contact, the pre-trained model processes live video feed data. It evaluates the eye regions and head pose for each frame, classifying the frame as either indicating eye contact (`True`) or not (`False`).

#### Real-Time Data Extraction from Camera Feed

- Utilizing **MediaPipe's Face Mesh**, the system captures real-time video feed through a camera using **OpenCV**. For each frame, it detects facial landmarks, focusing particularly on the regions around the eyes.
- Based on these landmarks, the system extracts the regions corresponding to the left and right eyes. This involves calculating the position of the eyes within the frame and cropping these areas for further analysis.
- Additionally, the system computes head pose data by estimating the orientation of the head in three-dimensional space, providing contextual information that can enhance the accuracy of eye contact detection.

#### Data Collection and Processing

- A **data collection mechanism** is implemented to gather training data. This mechanism captures images of the eyes along with head pose information while allowing users to indicate through keyboard inputs (e.g., pressing the spacebar) whether eye contact is being made. This labeled data is essential for training the machine learning model.
- The images undergo preprocessing, which includes resizing, grayscale conversion, tensor transformation, and normalization, to ensure they are in the correct format for model input.

#### Data Labels

- Images in the data directory have the label as their file name: **ID_Class_headDirection1_headDirection2_lipsDistance**
  - Class: (Bool) True/False (Eye Contact / No Contact)
  - headDirection1 and 2: (Float) Points of the vector which indicates in which direction the head points
  - lipsDistance: (Float) Space between the lips for further use of the model to detect if the person is speaking to the camera (not implemented yet)

#### User Interface and Interaction

- The system provides a simple interface template for understanding how to adapt and run the eye contact detection model. Users can start or stop data collection and toggle the eye contact label during data collection via keyboard inputs.

### Performance Enhancements

- **High-Resolution Camera**: Given the model's reliance on detailed pixel data around the eyes, using a high-resolution camera significantly improves detection accuracy.
- **Tuning Model Parameters**: Experimenting with layer sizes, activation functions, and other model parameters can lead to substantial improvements.
- **Omitting Head Pose Data**: Although including head pose data (representing the head's tilt in 3D space) can enrich the model's input, its dominant influence might sometimes overshadow other features. Removing this data can enhance predictions over longer distances and simplify the model's inputs.
- **Using CUDA/OPENCL/METAL** to enhance the efficiency and speed of training and inferencing the model

## Installation

To set up the project environment, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License, offering wide accessibility and flexibility for further development and application.

