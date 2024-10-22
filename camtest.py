import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the pre-trained Haar Cascade classifier for face detection
cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# Function to load the CNN model
def load_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the CNN model
input_shape = (150, 150, 1)  # Define input shape based on your image size
cnn_model = load_cnn_model(input_shape)
cnn_model.load_weights("skull_detection_model.h5")

# Function to process each frame from the camera
def capture_and_process():
    # Create a VideoCapture object to capture frames from the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_region = gray[y:y+h, x:x+w]
            # Resize and preprocess the face region for CNN input
            face_resized = cv2.resize(face_region, (input_shape[0], input_shape[1]))
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)
            # Predict using the CNN model
            prediction = cnn_model.predict(face_resized)
            # Convert prediction to class label
            class_label = "Skull" if prediction > 0.5 else "Not Skull"
            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Annotate the frame with the class label
            cv2.putText(frame, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow('Skull Detection', frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function to capture and process frames from the camera
capture_and_process()
