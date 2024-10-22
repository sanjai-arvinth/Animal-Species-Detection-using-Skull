import cv2
import os

def convert_mov_to_jpg(input_file, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the MOV file
    cap = cv2.VideoCapture(input_file)
    
    # Get the frame rate of the video
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the frame extraction interval (for one frame per second)
    frame_interval = frame_rate
    
    # Initialize frame count
    frame_count = 0
    
    # Read the first frame
    success, frame = cap.read()

    # Loop through all frames
    while success:
        # Check if the current frame is within the extraction interval
        if frame_count % frame_interval == 0:
            # Construct the output file path
            output_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")

            # Save the frame as a JPG file
            cv2.imwrite(output_file, frame)
        
        # Read the next frame
        success, frame = cap.read()

        # Increment frame count
        frame_count += 1
    
    # Release the VideoCapture
    cap.release()

# Folder containing MOV files
input_folder = r"F:\Project\Skull MOV Previews"

# Process each MOV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".mov"):
        # Construct input and output paths
        input_file = os.path.join(input_folder, filename)
        output_folder = os.path.join(input_folder, os.path.splitext(filename)[0])

        # Convert MOV to JPG
        convert_mov_to_jpg(input_file, output_folder)

print("Conversion completed.")
