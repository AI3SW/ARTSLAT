from pytube import YouTube
import os
import glob
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import csv
import traceback
    
class handDetector():
    def __init__(self, mode=True, maxHands = 2, detectionCon = 0.5, trackCon=0.5):
        self.mode = mode  # Object variable of mode = user inputted value, mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Must convert from BGR to RGB before processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img) # Allows this vairable to be used in any function

        flag = False
        if self.results.multi_hand_landmarks:
            flag = True
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                           self.mpHands.HAND_CONNECTIONS)
        return img, flag

    def findPosition(self, img, right_landmark, left_landmark, basecoord=None):
        # DID NOT FLIP HANDS SO JUST LET LEFT HAND (results.multihandness) == RIGHT (actual)

        left_flag, right_flag = False, False # actual, not processed ie. right(actly) = left

        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for idx, hand_handedness in enumerate(self.results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                label = handedness_dict['classification'][0]['label']

                if label == 'Left': #right actually
                    right_flag = True
                    right_landmark = []  #reset right landmark


                    targethand = self.results.multi_hand_landmarks[idx]
                    for id, lm in enumerate(targethand.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        right_landmark.extend([cx, cy]) # already flat

                    if basecoord == None:
                        base_x, base_y = right_landmark[0], right_landmark[1]
                        thumb_x, thumb_y = right_landmark[2], right_landmark[3]
                        scale_factor = (((thumb_x - base_x) ** 2) + ((thumb_y - base_y) ** 2)) ** 0.5

                        basecoord = [right_landmark[0],right_landmark[1], scale_factor]
                        basecoord[2] = max(scale_factor, 0.01)

                    else:
                        base_x, base_y, scale_factor = basecoord

                    for num in range(len(left_landmark)):
                        if num % 2 == 0:
                            right_landmark[num] -= base_x
                            right_landmark[num] /= scale_factor

                        else:
                            right_landmark[num] -= base_y
                            right_landmark[num] /= scale_factor



                if label == 'Right':  # left actually
                    left_flag = True
                    left_landmark = []  # reset left list

                    if basecoord == None:
                        base_x, base_y, scale_factor = 0,0,0.001

                    else:
                        base_x, base_y, scale_factor = basecoord

                    targethand = self.results.multi_hand_landmarks[idx]
                    for id, lm in enumerate(targethand.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        left_landmark.extend([cx, cy])

                    for num in range(len(left_landmark)):
                        if num % 2 == 0:
                            left_landmark[num] -= base_x
                            left_landmark[num] /= scale_factor

                        else:
                            left_landmark[num] -= base_y
                            left_landmark[num] /= scale_factor
            

            # END OF NORMALISATION


        return right_landmark, left_landmark, right_flag, left_flag, basecoord


detector = handDetector(mode = True)



class FrameExtractor():
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        
        
        
    def extract_landmarks(self, boundingbox, starttime, endtime, every_x_frame=1):
        """
        the boudy bounding box of the signer such as [y0, x0, y1, x1] where (x0, y0) is up-left corner and (x1,y1) is bottom-right corner. All the values are normalized (between zero and one) according to width and height.

        """
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
            
        '''
        HOW TO DEAL WITH 2 HANDS!!!!!!!!!!!
        NEED TO CHANGE HOW ITS NORMALISED (distance from an original point)
        '''
        
        # y0,x1 is max; y1,x0 is min
        width, height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        y0_pcnt, x0_pcnt, y1_pcnt, x1_pcnt = boundingbox
        y0, y1 = int(y0_pcnt*height), int(y1_pcnt*height)
        x0, x1 = int(x0_pcnt*width), int(x1_pcnt*width)
        
        
        starttime *= 1000 # convert to miliseconds
        endtime *= 1000

        self.vid_cap.set(cv2.CAP_PROP_POS_MSEC, starttime)
        
        frame_cnt = 0
        left_landmark = [[0,0] for i in range(14)]
        right_landmark = [[0,0] for i in range(14)]
        hand_landmark = []
        basecoord = None

        while self.vid_cap.isOpened():
            
            success,img = self.vid_cap.read() 

            if not success:
                break
            
            # y0,x1 is max; y1,x0 is min
            img = img[y0:y1, x0:x1]

            img,flag = detector.findHands(img,draw=True)

            # feed into findpos baselandmark, if ==default(which is none), if right hand detected, righthand = base
            right_landmark, left_landmark, right_flag, left_flag, basecoord = detector.findPosition(img, right_landmark,
                                                                                                    left_landmark, basecoord)

            right_landmark.extend(left_landmark)

            frame_landmark = right_landmark

            hand_landmark.append(frame_landmark)

            frame_cnt += 1


            if self.vid_cap.get(cv2.CAP_PROP_POS_MSEC) >= endtime:
                break
        
        self.vid_cap.release()
        cv2.destroyAllWindows()
        
        return hand_landmark

    
class VideoDownloader:
    def __init__(self, id):
        self.id = id # without https://www.youtube.com/watch?v=
        self.video_url = 'https://www.youtube.com/watch?v=' + id # Video ID should be a string
        
        
    def download(self, destFolder = './video_storage'):
        # '''
        # REMOVE ALL VIDEOS FROM DESTFOLDER
        # '''
        # files = glob.glob(destFolder + '/*')
        # for f in files:
        #     os.remove(f)


        video = YouTube(self.video_url)
        video = video.streams.filter(file_extension = "mp4")
        video.get_highest_resolution().download(output_path = destFolder,
                                                filename = self.id)
        
if __name__ == "__main__":
    import json

    with open(os.path.join(os.getcwd(), 'ms_asl/MSASL_train.json')) as f:
        data = json.load(f)

    counter = 0
    for video in data:

        if counter == 2:
            break

        starttime = video['start_time']
        endtime = video['end_time']
        label = video['label']
        bbox = video['box']
        url = video['url']
        url_id = url[url.find('watch?v=')+8:]

        try:
            alrDWL = False
            for file in os.listdir(os.path.join(os.getcwd(), 'video_storage')):
                if file == url_id+'.mp4':
                    alrDWL = True

            if not alrDWL:
                VD = VideoDownloader(url_id)
                VD.download()

            vidpath = os.path.join(os.getcwd(), 'video_storage', url_id+'.mp4')
            frameextractor = FrameExtractor(vidpath)
            hand_landmark = frameextractor.extract_landmarks(bbox, starttime, endtime)

        except Exception as e:
            str = traceback.print_exc()
            print(str)
            continue

        with open('landmark.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if len(hand_landmark) != 0:
                flat_landmarks = [item for sublist in hand_landmark for item in sublist]
                writtenlist = [label] + flat_landmarks
                spamwriter.writerow(writtenlist)

        print("Counter:", counter)
        counter +=1
