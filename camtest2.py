import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the pre-trained Haar Cascade classifier for face detection
cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# Specify the folder containing the training images and their respective classes
training_folder = r"F:\Project\PROJECT\train"

# Function to extract features from images
def extract_features(image):
    # You need to implement feature extraction suitable for your images
    # For example, you can use Histogram of Oriented Gradients (HOG) or other techniques
    # Here, we'll use a simple method of resizing the image and flattening it
    resized_image = cv2.resize(image, (100, 100))
    return resized_image.flatten()

# Function to compare the captured camera output to the images in the training folder using SVM
def compare_to_training_folder(image, training_folder):
    best_match = None
    min_mse = float('inf')

    # Create empty lists to store features and labels
    features = []
    labels = []

    # Iterate over the folders in the training directory
    for class_idx, class_name in enumerate(os.listdir(training_folder)):
        class_folder = os.path.join(training_folder, class_name)
        # Iterate over the files in each class folder
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            # Load the image from the training folder and convert it to grayscale
            train_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Extract features and append to the features list
            features.append(extract_features(train_img))
            # Append label for the corresponding class
            labels.append(class_idx)

    # Convert lists to arrays
    features = np.array(features)
    labels = np.array(labels)

    # Create SVM model
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))

    # Train SVM model
    svm_model.fit(features, labels)

    # Extract features from the captured image
    captured_features = extract_features(image)

    # Predict the class of the captured image using SVM model
    predicted_class = svm_model.predict([captured_features])

    # Retrieve the class name based on the predicted class index
    for class_idx, class_name in enumerate(os.listdir(training_folder)):
        if class_idx == predicted_class:
            best_match = class_name
            break

    return best_match

# Function to capture a grayscale still image from the camera
def capture_still_image():
    # Create a VideoCapture object to capture frames from the camera
    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return None
    
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    # Check if the frame is captured successfully
    if not ret:
        print("Error: Unable to capture frame.")
        return None
    
    # Convert the captured frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Release the VideoCapture object
    cap.release()
    
    return gray_frame


# Capture a still image
still_image = capture_still_image()

if still_image is not None:
    # Compare the captured image to the training folder using SVM
    best_match = compare_to_training_folder(still_image, training_folder)
    print("Best match:", best_match)
else:
    print("Error: No image captured.")
