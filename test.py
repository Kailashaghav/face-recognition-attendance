from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pyttsx3

# ---------------- TEXT TO SPEECH ----------------
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- LOAD CAMERA ----------------
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Camera not working")
    exit()

# ---------------- LOAD FACE DETECTOR ----------------
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# ---------------- LOAD DATA ----------------
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)
print('Labels length --> ', len(LABELS))

# ✅ FIX: Match lengths
if len(FACES) != len(LABELS):
    print("Fixing label mismatch...")
    min_len = min(len(FACES), len(LABELS))
    FACES = FACES[:min_len]
    LABELS = LABELS[:min_len]

# ---------------- TRAIN MODEL ----------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# ---------------- BACKGROUND ----------------
imgBackground = cv2.imread("/Users/kailashaghav/FACE RECOGNITION/backgroud.png")

# ✅ If background missing → create blank
if imgBackground is None:
    print("Background image not found, using plain background")
    imgBackground = np.zeros((600, 800, 3), dtype=np.uint8)

# ---------------- ATTENDANCE SETUP ----------------
COL_NAMES = ['NAME', 'TIME']

if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = video.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    attendance = None

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        exist = os.path.isfile(f"Attendance/Attendance_{date}.csv")

        # Draw box + name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        attendance = [str(output[0]), str(timestamp)]

    # ✅ Resize frame to match background
    frame_resized = cv2.resize(frame, (800, 600))
    imgBackground[0:600, 0:800] = frame_resized

    cv2.imshow("Face Recognition Attendance", imgBackground)

    key = cv2.waitKey(1)

    # ---------------- SAVE ATTENDANCE ----------------
    if key == ord('o') and attendance is not None:
        speak("Attendance Taken")

        if exist:
            with open(f"Attendance/Attendance_{date}.csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open(f"Attendance/Attendance_{date}.csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

        print("Attendance Saved:", attendance)
        time.sleep(2)

    # ---------------- EXIT ----------------
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()