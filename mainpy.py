import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from threading import Thread

# Function to upload an image using a file dialog
def upload_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to select the image file
    file_path = filedialog.askopenfilename()
    return file_path

# Function to resize an image
def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

# Function to find the best match for an input image within a source folder
def find_best_match(input_image, source_folder):
    # Initialize variables to keep track of the best match
    best_match_score = float('-inf')
    best_match_image = None
    best_match_class = None

    # Load the input image
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Iterate over each subfolder in the source folder
    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)
        
        # Check if the item in the source folder is a directory
        if os.path.isdir(subfolder_path):
            # Check if the directory contains image files
            image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if image_files:
                # Iterate over the images in the subfolder
                for filename in image_files:
                    image_path = os.path.join(subfolder_path, filename)

                    # Load the training image
                    train_image = cv2.imread(image_path)

                    # Resize the training image
                    resized_train_image = resize_image(train_image, input_image.shape[1], input_image.shape[0])

                    # Convert both images to grayscale
                    train_image_gray = cv2.cvtColor(resized_train_image, cv2.COLOR_BGR2GRAY)

                    # Perform template matching
                    result = cv2.matchTemplate(input_image_gray, train_image_gray, cv2.TM_CCOEFF_NORMED)

                    # Get the maximum similarity score and location
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    # Check if the current match is better than the previous best match
                    if max_val > best_match_score:
                        best_match_score = max_val
                        best_match_image = train_image
                        best_match_class = subfolder  # Update the best match class

    return best_match_image, best_match_class

# Function to process the image in a separate thread
def process_image():
    # Upload the input image
    input_image_path = upload_image()
    if input_image_path:
        input_image = cv2.imread(input_image_path)

        # Specify the source folder containing the training images
        source_folder = r"F:\Project\PROJECT"

        # Find the best match for the input image
        best_match_image, best_match_class = find_best_match(input_image, source_folder)

        # Display the best match class and image
        if best_match_image is not None and best_match_class is not None:
            best_match_label.config(text="Best match class: " + best_match_class)

            # Convert OpenCV image to PIL format for display in Tkinter
            best_match_image_rgb = cv2.cvtColor(best_match_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(best_match_image_rgb)
            pil_image.thumbnail((300, 300))  # Resize the image if needed
            img = ImageTk.PhotoImage(image=pil_image)
            image_label.config(image=img)
            image_label.image = img  # Keep a reference to the image to prevent garbage collection
        else:
            best_match_label.config(text="No match found.")
    else:
        best_match_label.config(text="No image selected.")

# Main function
def main():
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("ANIMAL SPECIES DETECTION USING SKULL")
    root.geometry("1200x1000")  # Set the window size

    # Set background color
    root.config(bg="#f0f0f0")

    # Create and position the widgets
    title_label = tk.Label(root, text="ANIMAL SPECIES DETECTION USING SKULL", font=("BahnSchrift", 20, "bold"), bg="#f0f0f0")
    title_label.pack(pady=20)

    upload_button = tk.Button(root, text="Upload Image", command=Thread(target=process_image).start)
    upload_button.pack(pady=10)

    global best_match_label
    best_match_label = tk.Label(root, text="", bg="#f0f0f0")
    best_match_label.pack(pady=10)

    global image_label
    image_label = tk.Label(root, bg="#f0f0f0")
    image_label.pack()

    # Run the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
