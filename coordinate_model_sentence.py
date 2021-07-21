import time
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

########### IMPORTANT = PIP INSTALL AUTOCORRECT
import autocorrect

from tensorflow.keras.models import load_model

from handdetector import handDetector
from normalise_landmarks import normalise_landmarks


try:
    model = load_model('./model.h5')
except:
    raise Exception("Cannot load model")



classconversion = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
                   8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
                   15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
                   22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}


def main():
    pTime = time.time()
    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionCon=0.7)

    spell = autocorrect.Speller(lang='en')

    avg_of_frame = 10
    counter = 1
    confidence = 0.7

    printed = ''

    startx = 500
    starty = 70
    recth = 40
    rectw = 80

    rolling = np.zeros((avg_of_frame,29))

    while True:
        success, img = cap.read()
        height, width, channels = img.shape # BAD BAD CODING DO NOT CONTINUE
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detector.findHands(img,draw=True)
        landmarks = detector.findPosition(img)

        #RESET BOX
        cv2.rectangle(img, (startx, starty), (startx + rectw, starty + recth), (128, 0, 0), -1)

        if len(landmarks) != 0:

            # RESET BUTTON
            fingers = detector.fingersUp()
            print(fingers)
            print(landmarks[8][0], landmarks[8][1])

            if fingers[1] == 1:
                if startx <= landmarks[8][0] <= (startx + rectw):
                    if starty <= landmarks[8][1] <= (startx + recth):
                        print('yes')
                        printed = ''
            # END RESET BUTTON


            try:
                normalised_landmarks = normalise_landmarks(landmarks)
            except:
                continue
            flat_landmarks  = [item for sublist in normalised_landmarks for item in sublist]
            X = np.array(flat_landmarks)
            X = X.reshape(1,42)

            predClass = model(X)
            predClass = predClass.numpy()


            # Rolling Average Implementation
            for id, element in enumerate(predClass[0]):
                rolling[counter,id] = element

            average = np.mean(rolling, axis=0)

            maxprob = np.amax(average)
            index = np.where(average == maxprob)
            letter = classconversion[index[0][0]] if maxprob>=confidence else ''

            counter = 0 if counter == avg_of_frame-1 else counter+1
            # End rolling avg implementation


            # Add correct letters NOT PREV LETTER INDEX -1 INSTEAD
            if letter == 'space':
                letter = ' '

            if len(printed) != 0 and printed[-1] != ' ':
                printed = spell(printed)

            if len(printed) != 0 and printed[-1] == letter: # Do not print same char again
                pass
            elif letter == 'del':
                printed = printed[:-1]
            else:
                printed += letter


        print(printed)
        cv2.putText(img, str(printed), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # FPS START --------------
        # cTime = time.time()
        # fps = 1 / (cTime-pTime)
        # pTime = cTime

        #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        # FPS END -----------------

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()