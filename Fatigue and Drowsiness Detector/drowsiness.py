import cv2
import dlib
from scipy.spatial import distance as dist

# Initialize Dlib's face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r"C:\Users\Acer\Desktop\Python-Face Tracking\shape_predictor_68_face_landmarks.dat")

def calculate_eye_aspect_ratio(eye):
    # Compute distances between vertical landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute distance between horizontal landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Calculate Eye Aspect Ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

# EAR threshold for drowsiness detection
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
frame_counter = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        # Get coordinates for eyes
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Calculate EAR for both eyes
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Check for drowsiness
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            frame_counter = 0

        # Draw eyes
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()