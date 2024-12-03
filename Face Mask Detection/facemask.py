import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import datetime
import os
import warnings

# Suppress TensorFlow and Keras logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses all but critical logs
warnings.filterwarnings("ignore")  # Suppress other warnings

# Model definition
model = Sequential([
    Input(shape=(150, 150, 3)),  # Explicitly define input shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training and test datasets
training_set = train_datagen.flow_from_directory(
    r'C:\Users\Acer\Desktop\Python-Face Tracking\FaceMaskDetector\train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    r'C:\Users\Acer\Desktop\Python-Face Tracking\FaceMaskDetector\test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# Train the model
model_saved = model.fit(
    training_set,
    epochs=10,
    validation_data=test_set
)

# Save the trained model
model.save(r'C:\Users\Acer\Desktop\Python-Face Tracking\FaceMaskDetector\mymodel.h5')

# Load the trained model for testing
mymodel = load_model(r'C:\Users\Acer\Desktop\Python-Face Tracking\FaceMaskDetector\mymodel.h5')

# Test on a single image
test_image = tf.keras.utils.load_img(
    r'C:\Users\Acer\Desktop\Python-Face Tracking\FaceMaskDetector\test\with_mask\1-with-mask.jpg',
    target_size=(150, 150)
)
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
print("Prediction:", "Mask" if mymodel.predict(test_image)[0][0] < 0.5 else "No Mask")

# Real-time face mask detection using webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        face_img = img[y:y + h, x:x + w]
        cv2.imwrite('temp.jpg', face_img)
        
        # Load and preprocess the face image
        test_image = tf.keras.utils.load_img('temp.jpg', target_size=(150, 150))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        pred = mymodel.predict(test_image)[0][0]

        # Draw rectangles and predictions
        label = 'NO MASK' if pred >= 0.5 else 'MASK'
        color = (0, 0, 255) if pred >= 0.5 else (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display date and time
    datet = str(datetime.datetime.now())
    cv2.putText(img, datet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Mask Detector', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q'
        break

cap.release()
cv2.destroyAllWindows()
