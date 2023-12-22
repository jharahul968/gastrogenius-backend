""" This is the flask application which 
    runs the ai model for Polyps Detection"""
import io
import os
import base64
from threading import Thread
from PIL import Image
from flask import Flask,jsonify,request
import cv2
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from segmentation import get_yolov5

model = get_yolov5()

app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

STOP_EXTRACTION_FLAG = False
 #pylint: disable=global-statement

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
    """
    This will stop the current running thread by setting the flag to True
    """
    #pylint: disable=no-member
    global STOP_EXTRACTION_FLAG
    #pylint: enable=no-member
    STOP_EXTRACTION_FLAG = True

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
    """
    This function will start a thread for extracting and processing the frame
    and return an acknowlegement from main thread.
    """
    global STOP_EXTRACTION_FLAG
    STOP_EXTRACTION_FLAG = False

    video_path = request.data.decode('utf-8')

    if video_path:
        Thread(target=extract_frames_from_video, args=(video_path,)).start()
        return jsonify({"ACK": True})
    return jsonify({"ACK": False, "error": "No video file received."})


def extract_frames_from_video(video_path):
    """
    The below three lines are used to decode the base64 
    encoded frame back to numpy array format which is neccesary
    for AI server to process
    """
    #pylint: disable=no-member
    cap = cv2.VideoCapture(video_path)
    # pylint: enable=no-member
    while True:
        if STOP_EXTRACTION_FLAG:
            break

        ret, frame = cap.read()

        if not ret:
            break
        # # pylint: disable=no-member
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pylint: enable=no-member
        pil_image = Image.fromarray(rgb_frame)

        results = model(pil_image)
        results.render()# updates results.imgs with boxes and labels
        # Process images in parallel

        for img in results.ims:
            img_base64 = Image.fromarray(img)
            image_bytes = io.BytesIO()
            img_base64.save(image_bytes, format="jpeg")
            base64_string = base64.b64encode(
                image_bytes.getvalue()).decode("utf-8")

        socketio.emit("Processed_Frame", base64_string)

    cap.release()


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    socketio.run(app,
                 host='127.0.0.1', port = 8000,
                 allow_unsafe_werkzeug=True, debug=True)
