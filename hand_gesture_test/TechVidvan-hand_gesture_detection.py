import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import random

# Load model and class names
model = load_model('mp_hand_gesture')
classNames = open('gesture.names', 'r').read().split('\n')

# Specify the desired gestures directly in the code
desired_gestures = ["thumbs up", "thumbs down", "peace", "okay", "stop"]  # Replace with your desired gestures

# Filter the class names to include only the desired gestures
filtered_classNames = [gesture for gesture in classNames if gesture.lower() in [g.lower() for g in desired_gestures]]

# Initialize MediaPipe and webcam
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Set frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Define game parameters
total_game_time = 30  # Total time for the entire game in seconds

# Map gesture class names to image file paths
gesture_images = {
    "thumbs up": "C:\\Users\\Admin\\Desktop\\hand_gesture_test\\mp_hand_gesture\\thumbs up.jpg",
    "thumbs down": "C:\\Users\\Admin\\Desktop\\hand_gesture_test\\mp_hand_gesture\\thumbs down.jpg",
    "peace": "C:\\Users\\Admin\\Desktop\\hand_gesture_test\\mp_hand_gesture\\peace.jpg",
    "okay": "C:\\Users\\Admin\\Desktop\\hand_gesture_test\\mp_hand_gesture\\okay.jpg",
    "stop": "C:\\Users\\Admin\\Desktop\\hand_gesture_test\\mp_hand_gesture\\stop.jpg",
}

def generate_gesture_sequence(length):
    """Generates a random gesture sequence."""
    return random.sample(filtered_classNames, length)

def display_gesture_images(gesture_sequence):
    """Displays gesture images for the user to mimic."""
    for gesture in gesture_sequence:
        image = cv2.imread(gesture_images[gesture])
        image = cv2.resize(image, (640, 360))  # Resize image to fit screen
        cv2.imshow("Gesture to Mimic", image)
        cv2.waitKey(2000)  # Display each image for 2 seconds
        cv2.destroyAllWindows()

def start_game():
    """Begins the game loop, tracking player gestures and scores."""
    while True:
        gesture_sequence = generate_gesture_sequence(5)  # Generate a sequence of 5 gestures
        display_gesture_images(gesture_sequence)  # Display gesture images before starting the game
        time.sleep(1)  # Add a short delay before starting the game
        current_gesture_index = 0
        start_time = time.time()
        game_over = False
        while True:
            success, frame = cap.read()
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(framergb)
            className = ''

            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * frame.shape[1])
                        lmy = int(lm.y * frame.shape[0])
                        landmarks.append([lmx, lmy])

                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                    prediction = model.predict([landmarks])
                    classID = np.argmax(prediction)
                    className = classNames[classID]

            # Check if there are more gestures to check and if className matches
            if current_gesture_index < len(gesture_sequence) and className == gesture_sequence[current_gesture_index]:
                current_gesture_index += 1
                cv2.putText(frame, "Correct!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check for game completion or timeout
            if current_gesture_index == len(gesture_sequence) and not game_over:
                cv2.putText(frame, "Congratulations! Sequence Completed!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                game_over = True

            if time.time() - start_time > total_game_time:
                if current_gesture_index < len(gesture_sequence) and not game_over:
                    cv2.putText(frame, "Game Over! Time's up!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    game_over = True

            if game_over:
                cv2.putText(frame, "Press 'r' to restart or 'q' to quit", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Gesture Game", frame)
                key = cv2.waitKey(0)  # Wait for a key press
                if key == ord('r'):
                    break  # Restart the game
                elif key == ord('q'):
                    break  # Exit the game

            else:
                cv2.putText(frame, "Target: " + gesture_sequence[current_gesture_index], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Time Remaining: " + str(max(0, int(total_game_time - (time.time() - start_time)))), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Gesture Game", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r') and game_over:
                break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the game
start_game()
