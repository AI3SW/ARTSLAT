import time
import cv2
import numpy as np

import argparse
import socketio
import base64

from tensorflow.keras.models import load_model

from handdetector import handDetector
from normalise_landmarks import normalise_landmarks


try:
    model = load_model('model.h5')
except:
    raise Exception("Cannot load model")


sio = socketio.Client()

@sio.event
def connect():
    print('[INFO] Successfully connected to server.')


@sio.event
def connect_error():
    print('[INFO] Failed to connect to server.')


@sio.event
def disconnect():
    print('[INFO] Disconnected from server.')


class CVClient(object):
    def __init__(self, server_addr, stream_fps):
        self.server_addr = server_addr
        self.server_port = 5001
        self._stream_fps = stream_fps
        self._last_update_t = time.time()
        self._wait_t = (1/self._stream_fps)

    def setup(self):
        print('[INFO] Connecting to server http://{}:{}...'.format(
            self.server_addr, self.server_port))
        sio.connect(
                'http://{}:{}'.format(self.server_addr, self.server_port),
                transports=['websocket'],
                namespaces=['/cv'])
        time.sleep(1)
        return self

    def _convert_image_to_jpeg(self, image):
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def send_data(self, frame, letter):
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            cv2.resize(frame, (640,480)) #MIGHT BE WRONG WAY TO RESIZE

            sio.emit(
                    'cv2server',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'text': '<br />'.join(letter)
                    })

    def check_exit(self):
        pass

    def close(self):
        sio.disconnect()



classconversion = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
                   8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
                   15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
                   22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}


def main(camera, use_streamer, server_addr, stream_fps):
    pTime = time.time() #FPS Use
    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionCon=0.7)

    avg_of_frame = 5
    counter = 1
    confidence = 0.5
    letter = ''
    rolling = np.zeros((avg_of_frame,29))
    streamer = None

    try:
        streamer = CVClient(server_addr, stream_fps).setup()

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
                if letter == 'space':
                    letter = ''
                if letter == 'del':
                    letter == ''

            streamer.send_data(img, letter)

            if streamer.check_exit():
                break

    finally:
        if streamer is not None:
            streamer.close()
        print("Program Ending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Streamer')
    parser.add_argument(
        '--camera', type=int, default='0',
        help='The camera index to stream from.')
    parser.add_argument(
        '--use-streamer', action='store_true',
        help='Use the embedded streamer instead of connecting to the server.')
    parser.add_argument(
        '--server-addr', type=str, default='localhost',
        help='The IP address or hostname of the SocketIO server.')
    parser.add_argument(
        '--stream-fps', type=float, default=20.0,
        help='The rate to send frames to the server.')
    args = parser.parse_args()
    main(args.camera, args.use_streamer, args.server_addr, args.stream_fps)