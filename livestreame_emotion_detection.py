
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from deepface import DeepFace
from collections import Counter

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize a counter for emotion frequencies
emotion_counter = Counter()

# Open a connection to the camera (assuming camera index 0, change it if needed)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y+h, x:x+w]

        # Analyze the face for age and emotion
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Access the dominant emotion for each face
        dominant_emotion = result[0]['dominant_emotion']

        # Update the emotion counter
        emotion_counter[dominant_emotion] += 1

        # Draw bounding box around the face with emotion information
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Frame with Bounding Boxes and Emotion", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera capture object and close the window
cap.release()
cv2.destroyAllWindows()

# Print the emotion frequencies
print("Emotion Frequencies:")
for emotion, count in emotion_counter.items():
    print(f"{emotion}: {count}")
