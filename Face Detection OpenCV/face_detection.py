import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern
import pickle

# Initialize the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Helper function to compute LBP features
def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=24, R=8, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Function to load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            lbp_features = extract_lbp_features(img)
            images.append(lbp_features)
            labels.append(os.path.basename(folder))
    return images, labels

# Load face data and train the recognizer
def train_recognizer(data_folders):
    x_data = []
    y_data = []
    for folder in data_folders:
        images, labels = load_images_from_folder(folder)
        x_data.extend(images)
        y_data.extend(labels)
    
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(y_data)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(x_data, y_data)
    
    return model, label_encoder

# Only include directories with the relevant face images
def get_face_image_dirs(base_path):
    face_dirs = []
    for dir_name in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path) and os.listdir(dir_path):
            face_dirs.append(dir_path)
    return face_dirs

base_path = os.getcwd()
user_dirs = get_face_image_dirs(base_path)
recognizer, label_encoder = train_recognizer(user_dirs)

# Save the trained model and label encoder
with open('face_recognizer.pkl', 'wb') as f:
    pickle.dump((recognizer, label_encoder), f)

# Load the recognizer and label encoder
with open('face_recognizer.pkl', 'rb') as f:
    recognizer, label_encoder = pickle.load(f)

# Initialize the video capture (default camera is 0)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the program.")

# Main loop for face recognition
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    frame = cv2.flip(frame, 1)  # Mirror the image for better interaction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 7)  # Detect faces

    for x, y, w, h in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (150, 150))

        # Extract LBP features and predict the face
        lbp_features = extract_lbp_features(face_img_resized)
        lbp_features = lbp_features.reshape(1, -1)
        pred = recognizer.predict(lbp_features)
        label = label_encoder.inverse_transform(pred)[0]

        # Draw a rectangle around the detected face and put the label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video feed with annotations
    cv2.imshow('Face Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Exiting program.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()