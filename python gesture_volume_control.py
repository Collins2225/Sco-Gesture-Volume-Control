import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
from collections import deque

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize Windows Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# ========== NEW FEATURE 1: FPS Counter Setup ==========
# Variables to track FPS
prev_time = 0
fps = 0

# ========== NEW FEATURE 2: Smoothing Setup ==========
# Store last 5 distance measurements for smoothing
distance_buffer = deque(maxlen=5)

print("=== Enhanced Gesture Volume Control ===")
print("Features:")
print("✓ Real-time volume control")
print("✓ Visual volume bar")
print("✓ Smoothing for stable control")
print("✓ FPS counter")
print("\nInstructions:")
print("- Show your hand to the camera")
print("- Pinch thumb and index finger together to decrease volume")
print("- Move them apart to increase volume")
print("- Press 'q' to quit")
print("=======================================\n")


# ========== NEW FEATURE 3: Volume Bar Drawing Function ==========
def draw_volume_bar(img, vol_percentage):
    """
    Draws a vertical volume bar on the right side of the screen

    Parameters:
    - img: The image to draw on
    - vol_percentage: Current volume (0-100)
    """
    # Bar dimensions
    bar_x = 580  # X position (right side)
    bar_y = 100  # Y position (top)
    bar_width = 40
    bar_height = 300

    # Draw background (empty bar)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (50, 50, 50), -1)

    # Draw border
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (255, 255, 255), 3)

    # Calculate filled height based on volume percentage
    filled_height = int((vol_percentage / 100) * bar_height)

    # Color changes based on volume level
    if vol_percentage < 33:
        color = (0, 0, 255)  # Red for low volume
    elif vol_percentage < 66:
        color = (0, 255, 255)  # Yellow for medium volume
    else:
        color = (0, 255, 0)  # Green for high volume

    # Draw filled portion (from bottom up)
    if filled_height > 0:
        cv2.rectangle(img,
                      (bar_x, bar_y + bar_height - filled_height),
                      (bar_x + bar_width, bar_y + bar_height),
                      color, -1)

    # Draw volume percentage text next to bar
    cv2.putText(img, f'{int(vol_percentage)}%',
                (bar_x - 10, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)


while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture video")
        break

    # ========== FPS CALCULATION ==========
    # Calculate time difference between frames
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    # Calculate FPS (frames per second)
    if time_diff > 0:
        fps = 1 / time_diff

    # Flip image horizontally for mirror effect
    img = cv2.flip(img, 1)

    # Convert BGR to RGB (MediaPipe uses RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to detect hands
    results = hands.process(img_rgb)

    # Default volume percentage for display
    vol_percentage = 0

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get coordinates of thumb tip (landmark 4) and index finger tip (landmark 8)
            landmarks = hand_landmarks.landmark

            # Thumb tip
            thumb_x = int(landmarks[4].x * img.shape[1])
            thumb_y = int(landmarks[4].y * img.shape[0])

            # Index finger tip
            index_x = int(landmarks[8].x * img.shape[1])
            index_y = int(landmarks[8].y * img.shape[0])

            # Draw circles on thumb and index finger tips
            cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)

            # Draw line between thumb and index finger
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)

            # Calculate distance between thumb and index finger
            distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            # ========== SMOOTHING: Add distance to buffer ==========
            distance_buffer.append(distance)

            # Calculate average distance from buffer (smoothed value)
            smoothed_distance = sum(distance_buffer) / len(distance_buffer)

            # Draw circle at midpoint
            mid_x = (thumb_x + index_x) // 2
            mid_y = (thumb_y + index_y) // 2
            cv2.circle(img, (mid_x, mid_y), 10, (0, 255, 0), cv2.FILLED)

            # Convert SMOOTHED distance to volume
            # Distance range: typically 20-200 pixels
            # Volume range: min_vol to max_vol
            vol = np.interp(smoothed_distance, [20, 200], [min_vol, max_vol])
            vol_percentage = np.interp(smoothed_distance, [20, 200], [0, 100])

            # Set the system volume
            volume.SetMasterVolumeLevel(vol, None)

            # Visual feedback - change circle color based on distance
            if smoothed_distance < 50:
                cv2.circle(img, (mid_x, mid_y), 10, (0, 0, 255), cv2.FILLED)

            # Display raw and smoothed distance for comparison (debugging)
            cv2.putText(img, f'Raw: {int(distance)}px',
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 2)
            cv2.putText(img, f'Smooth: {int(smoothed_distance)}px',
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

    # ========== DISPLAY FPS ==========
    cv2.putText(img, f'FPS: {int(fps)}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 3)

    # ========== DRAW VOLUME BAR ==========
    draw_volume_bar(img, vol_percentage)

    # Display the image
    cv2.imshow('Enhanced Gesture Volume Control', img)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

