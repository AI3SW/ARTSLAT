import cv2
import time

# cap = cv2.VideoCapture(0)
# pTime = time.time()
# while True:
#     success,img = cap.read()
#
#     cTime = time.time()
#     fps = 1/(cTime - pTime)
#     pTime = cTime
#
#     cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
#
#     cv2.imshow('image',img)
#     cv2.waitKey(1)


"""
showcase mediapipe
"""

from handdetector import handDetector

detector = handDetector()
cap = cv2.VideoCapture(0)
pTime = time.time()

while True:
    success,img = cap.read()
    img = detector.findHands(img, draw=True)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow('image',img)
    cv2.waitKey(1)