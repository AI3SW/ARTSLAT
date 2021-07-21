import mediapipe as mp
import cv2
import os
import csv


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

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # Allows this vairable to be used in any function
        
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
            scale_factor = ( ((thumb_x-base_x)**2) + ((thumb_y-base_y)**2) )**0.5
            
            for num in range(len(landmarks)):
                
                landmarks[num][0] -= base_x
                landmarks[num][0] /= scale_factor
                
                landmarks[num][1] -= base_y
                landmarks[num][1] /= scale_factor
                
                
        return landmarks



if os.path.isdir('train/augmented/A/') is False:
    os.mkdir('train/augmented')
    
    for directory in os.listdir(os.path.join(os.getcwd(), './train/asl_alphabet_train/asl_alphabet_train')):
            os.mkdir(f'train/augmented/{directory}')
            
            

detector = handDetector(mode = True)


original_directory = os.path.join(os.getcwd(), './train/asl_alphabet_train/asl_alphabet_train')
augmented_directory = os.path.join(os.getcwd(), './train/augmented')

with open('landmark.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for directory in os.listdir(original_directory):
        for imagefile in os.listdir(os.path.join(original_directory, directory)):
            
            imagepath = os.path.join(original_directory, directory, imagefile)
            img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
            
            img,flag = detector.findHands(img,draw=True)
            
            if not flag and directory != 'nothing':  # If no hands detected
                continue
            
            #LAZY WRITING
            singlelistlandmarks = [directory]
            try:
                landmarks = detector.findPosition(img)
            except:
                continue
            
            for item in landmarks:
                for coord in item:
                    singlelistlandmarks.append(coord)
                    
            if directory != 'nothing':
                spamwriter.writerow(singlelistlandmarks)
            
            newpath = os.path.join(augmented_directory, directory)
            cv2.imwrite(os.path.join(newpath , imagefile), img)
    
