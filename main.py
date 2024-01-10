""" This is the flask application which 
    runs the ai model for Polyps Detection"""
import io
import os
import base64
import numpy as np
import time
from threading import Thread
from PIL import Image
from flask import Flask,jsonify,request,abort
import cv2
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from segmentation import get_yolov5

model = get_yolov5()

app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = 'uploads'
FEEDBACK_FOLDER = 'feedback'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FEEDBACK_FOLDER'] = FEEDBACK_FOLDER
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

PROCESS_THREAD = None
PAUSE_EXTRACTING_FLAG = False
STOP_EXTRACTION_FLAG = False
REVERSE_FRAME = False
FORWARD_FRAME = False
CURRENT_FRAME_INDEX = 0

FRAME_HEIGHT,  FRAME_WIDTH = 0,0
CURRENT_FRAME = None
CURRENT_LABELS = []

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

    return jsonify({'error': 'Invalid file format'}), 404

@app.route('/feedback', methods=['POST'])
@cross_origin()
def get_feedback():
    """
    This function is responsible for receiving the feedback
    and saving those feedbacks in appropriate format in the server
    """
    data = request.json
    boxes = data.get('boxes')
    count = int()
    count_file = os.path.join(app.config['FEEDBACK_FOLDER'], 'count_frames.txt')
    global CURRENT_FRAME

    if not os.path.exists(app.config['FEEDBACK_FOLDER']):
        os.makedirs(app.config['FEEDBACK_FOLDER'])
        os.makedirs(os.path.join(app.config['FEEDBACK_FOLDER'], 'images'))
        os.makedirs(os.path.join(app.config['FEEDBACK_FOLDER'], 'labels'))
       

        with open(count_file, 'w') as file:
            count += 1
            file.write(str(count))
            file.close()

    else:
        with open(count_file, 'r') as file:
            count = int(file.read())
            file.close()

        count += 1

        with open(count_file, 'w') as file:
            file.write(str(count))
            file.close()
    
    rgb_image = Image.fromarray(CURRENT_FRAME)
    footage_name = os.path.join(app.config['FEEDBACK_FOLDER'], 'images', f"{count}.jpg")
    rgb_image.save(footage_name)


    #Time for the labels of pictures

    
    for box in CURRENT_LABELS:
            box = box.tolist()
            for i in range(len(box)):
                center_x = round((box[i][0] + ((box[i][2] - box[i][0]) / 2)) / FRAME_WIDTH, 6) 
                center_y = round((box[i][1] + ((box[i][3] - box[i][1]) / 2)) / FRAME_HEIGHT, 6)
                center_width = round((box[i][2] - box[i][0]) / FRAME_WIDTH, 6)
                center_height =  round((box[i][3] - box[i][1]) / FRAME_HEIGHT, 6)
                label = int(box[i][5])

                labelling_data = f"{label} {center_x} {center_y} {center_width} {center_height}"
                
                label_filepath = os.path.join(app.config['FEEDBACK_FOLDER'], 'labels', f"{count}.txt")

                with open(label_filepath, 'a+') as file:
                    file.write(labelling_data + '\n')



    for i, box in enumerate(boxes):

        adjustment_factor = 1470 - ((FRAME_WIDTH * 502 / FRAME_HEIGHT)) #Don't know why this 200px
        center_x = round((box['x'] - adjustment_factor + (box['width'] / 2)) / (FRAME_WIDTH * 502 / FRAME_HEIGHT), 6)
        center_y = round((box['y'] - 80 + (box['height'] / 2)) / 502, 6)
        center_width = round((box['width']) / (FRAME_WIDTH * 502 / FRAME_HEIGHT), 6)
        center_height = round((box['height']) / 502, 6)
        
        label = int()
        if box['label'] == "Adenomatous":
            label = 2
        else:
            label = 0

        labelling_data = f"{label} {center_x} {center_y} {center_width} {center_height}"

        label_filepath = os.path.join(app.config['FEEDBACK_FOLDER'], 'labels', f"{count}.txt")

        with open(label_filepath, 'a+') as file:
            file.write(labelling_data + '\n')

    return jsonify({'message':'successful'})


@app.route('/start-session', methods=["POST"])
@cross_origin()
def start_session():
    """
    This function will start a thread for extracting and processing the frame
    and return an acknowlegement from main thread.
    """
    global FRAME_WIDTH, FRAME_HEIGHT, STOP_EXTRACTION_FLAG

    STOP_EXTRACTION_FLAG = False
    
    video_path = request.data.decode('utf-8')


    if video_path:

        cap = cv2.VideoCapture(video_path)

        FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Use CAP_PROP_FRAME_WIDTH to get the frame width
        FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        size_message = {
        "width":FRAME_WIDTH,
            "height":FRAME_HEIGHT
        }
 

        FRAME_PROCESSING_THREAD = Thread(target=extract_frames_from_video, args=(video_path,))
        FRAME_PROCESSING_THREAD.start()

        return jsonify({"ACK": True, "size":size_message})
    
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

    global CURRENT_FRAME_INDEX, FRAME_HEIGHT, FRAME_WIDTH

    CURRENT_FRAME_INDEX = 0
    # FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Use CAP_PROP_FRAME_WIDTH to get the frame width
    # FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # size_message = {
    #    "width":FRAME_WIDTH,
    #     "height":FRAME_HEIGHT
    # }

    # socketio.emit('size', size_message)

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
        global REVERSE_FRAME, FORWARD_FRAME, CURRENT_FRAME, CURRENT_LABELS


        if not ret:
                return
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        CURRENT_FRAME = rgb_frame.copy()
        # pylint: enable=no-member
        results = model(rgb_frame)
        results.render()# updates results.imgs with boxes and labels
        CURRENT_LABELS = results.pred

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

        CURRENT_FRAME_INDEX += 1
        render(cap)

    os.remove(video_path)
    cap.release()


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    socketio.run(app,
                 host='0.0.0.0', port = 8000,
                 allow_unsafe_werkzeug=True, debug=True)
