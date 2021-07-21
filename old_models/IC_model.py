import time
import cv2
import mediapipe as mp
import numpy as np

from tensorflow.keras.models import load_model

from handdetector import handDetector


try:
    model = load_model('ASL.h5')
except:
    raise Exception("Cannot load model")

augm = 30
imgsz = 64


classconversion = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
                   8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
                   15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
                   22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}


def main():
    pTime = time.time()
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    counter = 1
    string = 'Start'

    while True:
        success, img = cap.read()
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detector.findHands(img,draw=True)
        landmarks = detector.findPosition(img)
        if len(landmarks) != 0:
             print(landmarks[4])
        mxl = detector.findmixmax(img, augm=augm)

        # DRAW BOUNDING BOX
        if len(mxl) != 0:
            cv2.rectangle(img, (mxl[0], mxl[2]+mxl[4]+2*augm), (mxl[0]+mxl[4]+2*augm, mxl[2]), (255, 0, 0), 2)

        if len(mxl) != 0:
            X = img[mxl[2]:mxl[2]+mxl[4]+2*augm, mxl[0]:mxl[0]+mxl[4]+2*augm]
            #cv2.imshow("cropped",x)
            X = cv2.resize(X, (imgsz,imgsz))
            X = np.asarray(X).reshape((-1,imgsz,imgsz,3))
            print(X.shape)
            pred = model.predict(x=X, verbose = 1)
            index = np.where(pred == np.amax(pred))
            predLetter = classconversion[index[1][0]]
            counter+=1
            #print(predLetter)
            #if counter % 4 == 0:
            string = predLetter
            #    counter = 1

        cv2.putText(img, str(string), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # FPS START --------------
        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime

        #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        # FPS END -----------------

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()