import cv2
import mediapipe as mp
#from google.protobuf.json_format import MessageToDict

class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon=0.5):
        self.mode = mode  # Object variable of mode = user inputted value, mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # Allows this vairable to be used in any function

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                           self.mpHands.HAND_CONNECTIONS)
                    # for idx, hand_handedness in enumerate(self.results.multi_handedness):
                    #     handedness_dict = MessageToDict(hand_handedness)
                    #     print(handedness_dict)

        return img

    def findPosition(self, img, handNo=0):

        self.landmarks = []

        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            targethand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(targethand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks.append([cx, cy])

        return self.landmarks

    def findmixmax(self, img, handNo=0, augm = 30):

        minmaxlist = []

        min_x, min_y = 9999,9999
        max_x, max_y = 1,1

        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            targethand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(targethand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

                min_x = min(cx, min_x)
                min_y = min(cy, min_y)
                max_x = max(cx, max_x)
                max_y = max(cy, max_y)

            maxlength = max(max_x-min_x,max_y-min_y, 1)

            # Align Square
            diffx, diffy = 0, 0
            diff = (max_x-min_x)-(max_y-min_y)

            if diff < 0:
                diffx = int(abs(diff)/2)
            else:
                diffy = int(diff/2)

            minmaxlist = [abs(min_x-augm-diffx), max_x,
                          abs(min_y-augm-diffy), max_y, maxlength]

        return minmaxlist

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.landmarks[self.tipIds[0]][0] > self.landmarks[self.tipIds[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.landmarks[self.tipIds[id]][1] < self.landmarks[self.tipIds[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)


            # totalFingers = fingers.count(1)

        return fingers
