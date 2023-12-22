""" This is the flask application which runs the ai model for Polyps Detection"""
import io
import os
import base64
from PIL import Image
import numpy as np
from flask import Flask, jsonify,request
import cv2
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from segmentation import get_yolov5
import time
from threading import Thread, Event
from redis import Redis

model = get_yolov5()

app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

stop_extraction_flag = False

#Subscribe to the 'channel_to_ai' Redis channel
@socketio.on('connect_with_frontend')
def handle_connect():
    """
    This code runs when a bidirectional connection is 
    established between frontend and this server.
    """
    print('Frontend connected to AI Server.')


@socketio.on("stop_thread")
@cross_origin()
def stop_thread():
    global stop_extraction_flag
    stop_extraction_flag = True

    return jsonify({'message':'Thread Stopped.'})


def allowed_file(filename):
    """
    This function defines the type of files allowed in the extraction
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/send-videos', methods=["POST"])
@cross_origin()
def extract_frames():
    """
    This function handles the incoming video file from the
    frontend and then saves it in uploads folder. It then returns
    an acknowledgement and the video path if the operation is successful.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return jsonify({'ack': True, 'filepath': filepath})

    return jsonify({'error': 'Invalid file format'})


@app.route('/start-session', methods=["POST"])
@cross_origin()
def start_session():

    global stop_extraction_flag
    stop_extraction_flag = False

    videoPath = request.data.decode('utf-8')

    if videoPath:
        Thread(target=extract_frames_from_video, args=(videoPath,)).start()
        return jsonify({"ACK": True})
    
    else:
         return jsonify({"ACK": False, "error": "No video file received."})


def extract_frames_from_video(videoPath):
    """
    The below three lines are used to decode the base64 
    encoded frame back to numpy array format which is neccesary
    for AI server to process
    """ 
    global stop_extraction_flag
    cap = cv2.VideoCapture(videoPath)
    # pylint: enable=no-member
    while True:
       
        if stop_extraction_flag:
            break

        ret, frame = cap.read()

        if not ret:
            break
      
        # # pylint: enable=no-member
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pylint: enable=no-member
        pil_image = Image.fromarray(rgb_frame)

        results = model(pil_image)
        results.render()  # updates results.imgs with boxes and labels
        # Process images in parallel

        for img in results.ims:
            img_base64 = Image.fromarray(img)
            image_bytes = io.BytesIO()
            img_base64.save(image_bytes, format="jpeg")
            base64_string = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        socketio.emit("Processed_Frame", base64_string)

    cap.release()


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    socketio.run(app, host='127.0.0.1', port = 8000, allow_unsafe_werkzeug=True, debug=True)
