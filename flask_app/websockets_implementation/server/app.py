from flask_socketio import SocketIO
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@socketio.on('connect', namespace='/web')
def connect_web():
    print('[INFO @ {}] Web client connected: {}'.format(datetime.now(), request.sid))


@socketio.on('disconnect', namespace='/web')
def disconnect_web():
    print('[INFO @ {}] Web client disconnected: {}'.format(datetime.now(), request.sid))


@socketio.on('connect', namespace='/cv')
def connect_cv():
    print('[INFO @ {}] CV client connected: {}'.format(datetime.now(), request.sid))


@socketio.on('disconnect', namespace='/cv')
def disconnect_cv():
    print('[INFO @ {}] CV client disconnected: {}'.format(datetime.now(), request.sid))


@socketio.on('cv2server')
def handle_cv_message(message):
    socketio.emit('server2web', message, namespace='/web')


if __name__ == "__main__":
    print('[INFO @ {}] Starting server at http://localhost:5001'.format(datetime.now()))
    socketio.run(app=app, host='0.0.0.0', port=5001)