import time
import cv2

import argparse
import socketio
import base64

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

    def send_data(self, frame):
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            cv2.resize(frame, (640,480))

            sio.emit(
                    'cv2server',
                    {
                        'image': self._convert_image_to_jpeg(frame)
                    })

    def check_exit(self):
        pass

    def close(self):
        sio.disconnect()


def main(use_streamer, server_addr, stream_fps):

    cap = cv2.VideoCapture(0)

    try:
        streamer = None
        streamer = CVClient(server_addr, stream_fps).setup()

        while True:
            success, img = cap.read()
            streamer.send_data(img)
            if streamer.check_exit():
                break

    finally:
        if streamer is not None:
            streamer.close()
        print("Program Ending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Streamer')
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
    main(args.use_streamer, args.server_addr, args.stream_fps)