import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./trained_model.pickle', 'rb'))
model = model_dict['model']

# Initialize the camera
cap = cv2.VideoCapture(0)  # Adjust the camera index if needed
if not cap.isOpened():
    raise RuntimeError("Camera not opened. Please check your camera setup.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Dictionary for class labels
labels_dict = {
    'A': 'A', 'B': 'B', 'Call': 'Call', 'Goodbye': 'Goodbye', 'Hello': 'Hello',
    'I': 'I', 'No': 'No', 'Thankyou': 'Thankyou', 'Yes': 'Yes', 'You': 'You'
}

# Main loop for video capture and processing
while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret or frame is None:
        print("Failed to capture a frame. Exiting loop.")
        break  # Exit if frame capture fails

    # Show the live feed
    cv2.imshow("Live Feed", frame)

    # Check for keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"):  # When 'p' is pressed
        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            hand_landmarks = results.multi_hand_landmarks[0]

            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            min_x = min(x_)
            min_y = min(y_)

            # Collect landmark data for prediction
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min(y_))

            # Make a prediction
            prediction = model.predict([np.asarray(data_aux)])

            # Fetch the predicted label from `labels_dict`
            predicted_character = labels_dict.get(prediction[0], "Unknown")

            # Draw bounding box and label on the frame
            H, W, _ = frame.shape
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Create a copy of the frame to draw on
            frame_with_label = frame.copy()
            cv2.rectangle(frame_with_label, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame_with_label, predicted_character,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 0, 0),
                        3, cv2.LINE_AA)

            # Display the frame with the bounding box and label
            cv2.imshow("Prediction", frame_with_label)

    # Exit when 'q' is pressed
    if key == ord("q"):
        break  # Exit loop when 'q' is pressed

# Release resources and close OpenCV windows
cap.release()
cv2.destroyAllWindows()  # Close all OpenCV windows
