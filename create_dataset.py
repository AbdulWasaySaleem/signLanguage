import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Define the root directory for your data
DATA_DIR = './Data'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  
    min_detection_confidence=0.5,  
    max_num_hands=2  
)

# Create a list to store the data and labels
dataset = []
labels = []

# Iterate through each subfolder in the data directory
for subfolder in sorted(os.listdir(DATA_DIR)):
    subfolder_path = os.path.join(DATA_DIR, subfolder)
    
    if os.path.isdir(subfolder_path):
        label = subfolder  # Name of the subfolder is used as the label
        
        # Iterate through each image in the subfolder
        for img_filename in os.listdir(subfolder_path):
            # Read the image
            img_path = os.path.join(subfolder_path, img_filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image to RGB
                results = hands.process(img_rgb) # Process the image with MediaPipe Hands
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Collect landmark data
                        data_aux = []
                        x_ = [landmark.x for landmark in hand_landmarks.landmark]
                        y_ = [landmark.y for landmark in hand_landmarks.landmark]
                        min_x = min(x_)
                        min_y = min(y_)
                        
                        for landmark in hand_landmarks.landmark:
                            data_aux.append(landmark.x - min_x)
                            data_aux.append(landmark.y - min(y_))
                        
                        # Add the data and the corresponding label to the dataset
                        dataset.append(data_aux)
                        labels.append(label)

# Save the dataset using pickle
with open('sign_language_dataset.pickle', 'wb') as f:
    pickle.dump({'data': dataset, 'labels': labels}, f)

print("Dataset created and saved successfully.")
