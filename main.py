import clf
import cv2
import numpy as np
import tkinter as tk
from tkinter import *


root = Tk()
root.title("app")
root.geometry("600x550")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def photo_face_detection():
    frame = cv2.imread('C:\Progs\group2.jpg')

    filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaling_factor = 1.25

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=1)
    eyes = eye_cascade.detectMultiScale(frame, scaleFactor= 1.05, minNeighbors=14)
    smile = smile_cascade.detectMultiScale(frame, scaleFactor= 2, minNeighbors=20)

    print(f"Found {len(faces)} faces!")
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    for (ax, ay, aw, ah) in eyes:
        cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (0, 0, 255), 1)
    for (bx, by, bw, bh) in smile:
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)

    cv2.imshow('Face', frame)
    cv2.waitKey(0)

def HOG_detection():

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector (cv2.HOGDescriptor_getDefaultPeopleDetector())
    cv2.startWindowThread()
    cap = cv2.VideoCapture('street_footage.mp4')

    while True:
        ret, frame1 = cap.read()
        frame1 = cv2.resize(frame1, (880, 560))
        gray_filter = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale (frame1, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xa, ya, xb, yb) in boxes:
            cv2.rectangle(frame1, (xa, ya), (xb, yb), (0, 255, 255), 1)
            cv2.imshow('Video', frame1)

        if (cv2.waitKey(1) & 0xFF==ord('q')):
            break
    cap.release()

def frontal_video_face_detection():
    cap = cv2.VideoCapture('face_footage.mp4')
    while True:
        ret, face_footage = cap.read()
        face_footage = cv2.resize(face_footage, (880, 560))
        face_gray_filter = cv2.cvtColor(face_footage, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(face_gray_filter, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(100, 100))
        for (x, y, w, h) in faces:
            cv2.rectangle(face_footage, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow('Face', face_footage)
        if (cv2.waitKey(1) & 0xFF==ord('q')):
            break
    face_footage.release()
    cv2.destroyAllWindows()

show_btn = tk.Button(root, text="Image face detection", command=lambda: photo_face_detection())
show_btn.pack(fill = BOTH, expand = True)

cut_btn = tk.Button(root, text="HOG people detection", command=lambda: HOG_detection())
cut_btn.pack(fill = BOTH, expand = True)

resize_btn = tk.Button(root, text="Video face detection", command=lambda: frontal_video_face_detection())
resize_btn.pack(fill = BOTH, expand = True)


root.mainloop()