import time
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

from handdetector import handDetector
from normalise_landmarks import normalise_landmarks


try:
    model = load_model('./model.h5')
except:
    raise Exception("Cannot load model")


# Converting class indices to the appropriate letter
classconversion = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
                   8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
                   15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
                   22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}


def main():
    # pTime = time.time() # FPS Implementation
    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionCon=0.7)

    avg_of_frame = 5
    counter = 1
    confidence = 0.5
    letter = 'Start'
    rolling = np.zeros((avg_of_frame,29))

    while True:
        success, img = cap.read()
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detector.findHands(img,draw=True)
        landmarks = detector.findPosition(img)

        if len(landmarks) != 0:
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

        cv2.putText(img, str(letter), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

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