import mediapipe as mp
import os
import cv2


class handDetector():
    def __init__(self, mode=True, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode  # Object variable of mode = user inputted value, mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        self.results = self.hands.process(img)  # Allows this vairable to be used in any function

        flag = False
        if self.results.multi_hand_landmarks:
            flag = True
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img, flag

    def findPosition(self, img, handNo=0):

        landmarks = []

        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            targethand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(targethand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])

            # NORMALISE :) !!!DOES NOT ACCOUNT FOR DEPTH
            base_x, base_y = landmarks[0][0], landmarks[0][1]
            thumb_x, thumb_y = landmarks[1][0], landmarks[1][1]
            scale_factor = (((thumb_x - base_x) ** 2) + ((thumb_y - base_y) ** 2)) ** 0.5

            for num in range(len(landmarks)):
                landmarks[num][0] -= base_x
                landmarks[num][0] /= scale_factor

                landmarks[num][1] -= base_y
                landmarks[num][1] /= scale_factor

        return landmarks


detector = handDetector(mode=True)



for item in os.listdir(os.path.join(os.getcwd(), 'video_storage')):
    if item[-3:] == 'jpg':
        impath = os.path.join(os.getcwd(), 'video_storage', item)
        img = cv2.imread(impath)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img,flag = detector.findHands(img, draw=True)
        cv2.imwrite(impath,img)
        #cv2.imshow('image', img)
        #cv2.waitKey()