import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Define the root directory for your data
DATA_DIR = './Data'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create a MediaPipe Hands object
hands = mp_hands.Hands(
    static_image_mode=True,  
    min_detection_confidence=0.5, 
    max_num_hands=2  
)

# Iterate through each subfolder in the data directory
for subfolder in sorted(os.listdir(DATA_DIR)):
    subfolder_path = os.path.join(DATA_DIR, subfolder)
    
    # Check if it's a valid directory
    if os.path.isdir(subfolder_path):
        # Get the first image in the subfolder
        first_image_path = sorted(os.listdir(subfolder_path))[0]  # Get the first image by name
        img_path = os.path.join(subfolder_path, first_image_path)
        
        # Read the image
        img = cv2.imread(img_path)
        
        if img is not None:
            # Convert the image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image with MediaPipe Hands
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the image
                    mp_drawing.draw_landmarks(
                        img_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                
                # Display the image with landmarks
                plt.figure()
                plt.imshow(img_rgb)
                plt.title(f"Landmarks for {subfolder}")
                plt.axis('off')  # Hide the axes for better visualization
                plt.show()


    
