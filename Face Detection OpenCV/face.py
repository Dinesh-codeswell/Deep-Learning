import cv2
import os
import time

# Ensure the correct path for the Haar cascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ask the user to enter their name and store it
user_name = input("Please enter your name: ")

# Create a directory to store the user's face data if it doesn't exist
if not os.path.exists(user_name):
    os.makedirs(user_name)

print(f"Hello, {user_name}! The face tracking system will start now.")

# Initialize the video capture (default camera is 0)
cap = cv2.VideoCapture(0)

start_time = time.time()
capture_duration = 15  # Capture duration in seconds
image_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    frame = cv2.flip(frame, 1)  # Mirror the image for better interaction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)  # Detect faces

    for x, y, w, h in faces:
        # Calculate the center of the detected face
        center_x, center_y = x + w // 2, y + h // 2

        # Save the captured face frames into the directory
        face_img = frame[y:y+h, x:x+w]
        face_filename = os.path.join(user_name, f"{user_name}_{image_count}.jpg")
        cv2.imwrite(face_filename, face_img)
        image_count += 1

        # Draw a circle at the center of the face
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Show the video feed with annotations
    cv2.imshow('Face Tracking', frame)
    
    # Exit after the specified duration
    if time.time() - start_time > capture_duration:
        print(f"Face tracking completed. {image_count} images saved.")
        break

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Exiting program.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()