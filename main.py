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
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

PROCESS_THREAD = None
PAUSE_EXTRACTING_FLAG = False
STOP_EXTRACTION_FLAG = False
REVERSE_FRAME = False
FORWARD_FRAME = False
CURRENT_FRAME_INDEX = 0
 #pylint: disable=global-statement

def allowed_file(filename):
    """
    This function defines the type of files allowed in the extraction
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@socketio.on("Reverse")
@cross_origin()
def reverse_frame():
    global REVERSE_FRAME, FORWARD_FRAME, PAUSE_EXTRACTING_FLAG
    
    REVERSE_FRAME = True
    FORWARD_FRAME = False
    PAUSE_EXTRACTING_FLAG = True
    
    return jsonify({"message":"Reversed"})


@socketio.on("Forward")
@cross_origin()
def reverse_frame():
    global FORWARD_FRAME, REVERSE_FRAME, PAUSE_EXTRACTING_FLAG
    
    REVERSE_FRAME = False
    FORWARD_FRAME = True
    PAUSE_EXTRACTING_FLAG = True

    return jsonify({"message":"Forwarded"})


@socketio.on("Pause")
@cross_origin()
def pause_session():
    """
    This function pauses the real time session
    """
    global PAUSE_EXTRACTING_FLAG
    PAUSE_EXTRACTING_FLAG = True

    return jsonify({"message":"Paused"})


@socketio.on("Unpause")
@cross_origin()
def unpause_session():
    """
    This function unpauses the real time session
    """
    global PAUSE_EXTRACTING_FLAG, REVERSE_FRAME, FORWARD_FRAME

    PAUSE_EXTRACTING_FLAG = False
    REVERSE_FRAME = False
    FORWARD_FRAME = False

    return jsonify({"message":"Unpaused"})


@socketio.on("stop_thread")
@cross_origin()
def stop_thread():
    """
    This will stop the current running thread by setting the flag to True
    """
    #pylint: disable=no-member
    global STOP_EXTRACTION_FLAG, PAUSE_EXTRACTING_FLAG, REVERSE_FRAME, FORWARD_FRAME
    #pylint: enable=no-member

    PAUSE_EXTRACTING_FLAG = False
    REVERSE_FRAME = False
    FORWARD_FRAME = False
    STOP_EXTRACTION_FLAG = True

    return jsonify({'message':'Thread Stopped.'})


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
    global STOP_EXTRACTION_FLAG, PROCESS_THREAD
    STOP_EXTRACTION_FLAG = False
    
    video_path = request.data.decode('utf-8')

    if video_path:
        PROCESS_THREAD = Thread(target=extract_frames_from_video,args=(video_path,))
        PROCESS_THREAD.start()
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    global CURRENT_FRAME_INDEX

    CURRENT_FRAME_INDEX = 0

    def convert_to_base64(img):
            """
            This sub function converts rgb image data
            into base64_string
            """
            img_base64 = Image.fromarray(img)
            image_bytes = io.BytesIO()
            img_base64.save(image_bytes, format="jpeg")
            base64_string = base64.b64encode(
                image_bytes.getvalue()).decode("utf-8")
            
            return base64_string
    
    def render(cap):
        """
        This sub function handles the rendering and 
        emiting of the frames to frontend service
        """
        ret, frame = cap.read()
        global REVERSE_FRAME, FORWARD_FRAME

        if not ret:
                return
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pylint: enable=no-member
        results = model(rgb_frame)
        results.render()# updates results.imgs with boxes and labels
        # Process images in parallel

        for img in results.ims:
            base64_string = convert_to_base64(img)

        socketio.emit("Processed_Frame", base64_string)

        if REVERSE_FRAME:
            REVERSE_FRAME = False
        if FORWARD_FRAME:
            FORWARD_FRAME = False
        
   
    # pylint: enable=no-member
    while True:

        if STOP_EXTRACTION_FLAG:
            break
        
        if PAUSE_EXTRACTING_FLAG:
            
            if REVERSE_FRAME:
                CURRENT_FRAME_INDEX = max(0, CURRENT_FRAME_INDEX - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, CURRENT_FRAME_INDEX)

                render(cap)
                continue

            elif FORWARD_FRAME:
                CURRENT_FRAME_INDEX = min(total_frames - 1, CURRENT_FRAME_INDEX + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, CURRENT_FRAME_INDEX)

                render(cap)
                continue

            else:
                continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, CURRENT_FRAME_INDEX)
        CURRENT_FRAME_INDEX += 1
        render(cap)
   
    cap.release()


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    socketio.run(app,
                 host='0.0.0.0', port = 8000,
                 allow_unsafe_werkzeug=True, debug=True)
