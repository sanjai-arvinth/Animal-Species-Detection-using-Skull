import cv2
import os
import numpy as np
import csv
from collections import Counter

# Load the pre-trained Haar Cascade classifier for face detection
cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# Specify the folder containing the training images and their respective classes
training_folder = r"F:\Project\PROJECT\train"

# Function to read the predicted class names from the CSV file
def read_predicted_classes(csv_file):
    predicted_classes = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            predicted_classes.extend(row)
    return predicted_classes

# Function to find the most frequently occurring classes
def find_most_frequent_classes(predicted_classes):
    class_counts = Counter(predicted_classes)
    most_common_classes = class_counts.most_common()
    return most_common_classes

# Function to compare the captured camera output to the images in the training folder
def compare_to_training_folder(image, training_folder):
    best_match_class = None
    best_match_score = float('inf')

    # Iterate over the folders in the training directory
    for class_name in os.listdir(training_folder):
        class_folder = os.path.join(training_folder, class_name)
        # Iterate over the files in each class folder
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            # Load the image from the training folder
            train_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Resize the training image to match the captured image
            train_img = cv2.resize(train_img, (image.shape[1], image.shape[0]))
            # Compute the Mean Squared Error (MSE) between the captured image and the training image
            mse = np.mean((train_img - image) ** 2)
            # Update the best match if the MSE is lower
            if mse < best_match_score:
                best_match_score = mse
                best_match_class = class_name

    return best_match_class

# Function to capture an image from the camera
def capture_image():
    # Create a VideoCapture object to capture frames from the camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        return None
    cap.release()
    cv2.destroyAllWindows()
    return frame

# Read the predicted class names from the CSV file
csv_file = r"F:\Project\predicted_classes.csv"
predicted_class_names = read_predicted_classes(csv_file)

# Find the most frequently occurring classes
most_frequent_classes = find_most_frequent_classes(predicted_class_names)

# Capture an image from the camera
captured_image = capture_image()

if captured_image is not None:
    # Display the captured image
    cv2.imshow("Captured Image", captured_image)
    cv2.waitKey(0)

    # Compare the captured image to the training folder
    best_match_class = compare_to_training_folder(captured_image, training_folder)
    print("Best Match Class:", best_match_class)
else:
    print("No image captured.")
