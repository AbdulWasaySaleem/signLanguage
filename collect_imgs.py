import os
import cv2

# Define the subcategories
subcategories = ["A", "B", "I", "Thankyou", "Hello", "Call", "Goodbye", "You", "Yes", "No"]

# Create the main data directory
DATA_DIR = './Data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Create subfolders for each subcategory
for subcategory in subcategories:
    subfolder_path = os.path.join(DATA_DIR, subcategory)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Adjust the webcam ID if needed

# Check if the webcam is working
if not cap.isOpened():
    raise RuntimeError("Webcam not opened. Please check connections or permissions.")

# Define capture parameters
capture_limit = 100
current_category_index = 0  # Index of the current subcategory

while True:
    # Check if we've captured for all subcategories
    if current_category_index >= len(subcategories):
        print("All categories have been captured. Exiting.")
        break

    # Display the instruction for the current subcategory
    subcategory = subcategories[current_category_index]
    print(f"Capturing images for '{subcategory}'.")

    capture_counter = 0
    capturing = False  # Flag to control capturing

    # Loop to capture the specified number of images for the current category
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the frame capture fails

        # Display the instruction to start capturing
        cv2.putText(
            frame,
            f"Capturing for '{subcategory}'. Press 's' to start.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # If capturing, save the images
        if capturing:
            subfolder_path = os.path.join(DATA_DIR, subcategory)
            filename = f"image_{capture_counter}.jpg"
            filepath = os.path.join(subfolder_path, filename)

            cv2.imwrite(filepath, frame)  # Save the image
            capture_counter += 1

            if capture_counter >= capture_limit:
                capturing = False  # Stop capturing
                current_category_index += 1  # Move to the next subcategory
                break  # Exit the inner loop once the capture limit is reached

        # Display the webcam feed
        cv2.imshow("Webcam Feed", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            capturing = True  # Start capturing

        if key == ord("q"):
            break  # Exit the outer loop to quit the script

    if key == ord("q"):
        break  # Exit the outer loop if 'q' is pressed

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
