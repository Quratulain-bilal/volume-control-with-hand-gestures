import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Volume control setup using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()  # (-65.25, 0.0, 0.03125)

# Variables for volume control
min_volume, max_volume = volume_range[0], volume_range[1]

# Function to display text on the frame
def display_text(frame, text, position, color):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural movement
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # If hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the position of key landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Calculate the distance between wrist and middle finger tip (for palm open/close)
            distance_wrist_middle = ((wrist.x - middle_finger_tip.x) ** 2 +
                                     (wrist.y - middle_finger_tip.y) ** 2) ** 0.5

            # Palm Open/Close Detection
            if distance_wrist_middle > 0.1:  # Threshold for palm open
                # Palm is open: Control volume
                # Map the distance to the volume range smoothly
                volume_level = np.interp(distance_wrist_middle, [0.1, 0.4], [min_volume, max_volume])
                volume.SetMasterVolumeLevel(volume_level, None)
                display_text(frame, f"Volume: {int(np.interp(volume_level, [min_volume, max_volume], [0, 100]))}%", (50, 100), (255, 0, 0))
            else:
                # Palm is closed: Check for thumb gestures
                # Thumb Up: Check if thumb tip is above index finger tip
                if thumb_tip.y < index_finger_tip.y:
                    display_text(frame, "Thumb Up", (50, 50), (0, 255, 0))  # Green text for "Thumb Up"

                # Thumb Down: Check if thumb tip is below index finger tip
                elif thumb_tip.y > index_finger_tip.y:
                    display_text(frame, "Thumb Down", (50, 50), (0, 0, 255))  # Red text for "Thumb Down"

    # Display the frame with hand landmarks
    cv2.imshow("Hand Gesture Control", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()