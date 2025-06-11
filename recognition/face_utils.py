import face_recognition
import cv2
# from django.core.files.base import ContentFile
import numpy as np
# import io
# from PIL import Image
import os
import face_recognition

# REMINDER TO CREATE A FUNCTION THAT TAKES THE ENCODINGS AND STORES THEM IN THE DATABASE UNDER THE INSTITUION

# def encode_face(image_path):
#     image = face_recognition.load_image_file(image_path)
#     encodings = face_recognition.face_encodings(image)

#     if not encodings:
#         return None # No faces

#     return encodings[0]

# def detect_faces_from_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     return faces

# Configuration
known_faces_dir = './uploads/faces/'
face_idcard_dir = './uploads/faces/cards/'
scale_factor = 0.25  # Reduced resolution for processing
tolerance = 0.5  # Lower = stricter matching
target_fps = 30  # Target frames per second
process_every_n_frames = 2  # Process every nth frame to reduce load

# === Load known faces ===
print("🔄 Loading known face images...")

# Cache for known faces and ID cards
known_face_encodings = []
known_face_names = []
id_card_cache = {}

# Preload all known faces and ID cards
for filename in os.listdir(known_faces_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    path = os.path.join(known_faces_dir, filename)
    image = cv2.imread(path)

    if image is None:
        print(f"⚠️ Could not read image: {path}")
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)

    if encodings:
        known_face_encodings.append(encodings[0])
        name = os.path.splitext(filename)[0].capitalize()
        known_face_names.append(name)

        # Preload ID card if exists
        idcard_path = os.path.join(face_idcard_dir, f'ID_{name}.jpg')
        if os.path.exists(idcard_path):
            id_card_cache[name] = cv2.imread(idcard_path)

        print(f"✅ Loaded: {name}")
    else:
        print(f"⚠️ No face detected in: {path}")

if not known_face_encodings:
    raise RuntimeError("❌ No known faces loaded. Please check your images.")

# === Start webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, target_fps)

print("📷 Starting webcam...")

frame_count = 0
face_locations = []
face_encodings = []
names = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ Failed to read from webcam.")
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)

    # Only process every nth frame to improve FPS
    if frame_count % process_every_n_frames == 0:
        # Resize and convert frame once
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Get face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Reset names for this frame
        names = []

        for face_encoding in face_encodings:
            # Use numpy for faster distance calculation
            distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
            matches = distances <= tolerance

            if any(matches):
                best_match_index = np.argmin(distances)
                name = known_face_names[best_match_index]
                names.append(name)
            else:
                names.append(None)

    # Draw results on the frame
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Scale back up face location
        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)

        if name:
            first_name = name.split(" ")[0]

            # Draw face box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, first_name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Show ID card from cache
            if name in id_card_cache:
                idcard = id_card_cache[name]
                face_width = right - left
                face_height = bottom - top

                if face_width > 0 and face_height > 0:
                    idcard_resized = cv2.resize(idcard, (face_width, face_height))
                    frame[top:bottom, right:right + face_width] = idcard_resized

    # Display FPS information
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()