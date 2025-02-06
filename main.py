import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


cap = cv2.VideoCapture(0)
scaling_factor = 1.25
frame = cv2.imread('C:\Progs\group2.jpg')
frame1 = cv2.imread('C:\Progs\group2.jpg')

filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
cv2.destroyAllWindows()


