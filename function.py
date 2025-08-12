import cv2
from deepface import DeepFace
import pygame
def analyze_emotion(frame):
    result = DeepFace.analyze(frame, actions=['emotion'])
    return result[0]['dominant_emotion']

def opencam():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    if face_cascade.empty():
        raise IOError("Failed to load Haar cascade XML file.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    frozen_frame = None
    capturing = True  # True = live feed, False = frozen

    while True:
        if capturing:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed, exiting.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(110, 110))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
            cv2.imshow("Face Detection", frame)
        else:
            cv2.imshow("Face Detection", frozen_frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC → exit
            break
        elif k == 32 and capturing:  # SPACE → freeze
            frozen_frame = frame.copy()
            capturing = False
            emotion = analyze_emotion(frozen_frame)
            song(emotion)
            cv2.putText(frozen_frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            print(f"Detected Emotion: {emotion}")    # show in terminal

            print("Frame frozen. Press ESC to exit.")

    cap.release()
    cv2.destroyAllWindows()

def song(emotion):
    pygame.mixer.init()
    if emotion == 'happy':
        pygame.mixer.music.load('happi.mp3')
    elif emotion == 'neutral':
        pygame.mixer.music.load('/home/prince/Song-recommder/sad.mp3')
    else:
        return
    pygame.mixer.music.play()
    

opencam()
