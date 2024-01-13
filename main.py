""" This is the flask application which 
    runs the ai model for Polyps Detection"""
import io
import os
import base64
from threading import Thread, Lock
from PIL import Image
from flask import Flask,jsonify,request, session
import cv2
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, join_room, leave_room, send
from segmentation import get_yolov5
from concurrent.futures import ThreadPoolExecutor
import random

model = get_yolov5()

# app = Flask(__name__, static_folder='./build', static_url_path='/')

thread_pool = ThreadPoolExecutor(max_workers=10) 
app = Flask(__name__)
app.secret_key = '__your_secret_key_-'
cors = CORS(app)

UPLOAD_FOLDER = 'uploads'
FEEDBACK_FOLDER = 'feedback'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FEEDBACK_FOLDER'] = FEEDBACK_FOLDER
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

users ={}
CURRENT_LABELS = []

def allowed_file(filename):
    """
    This function defines the type of files allowed in the extraction
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class UserSession:
    def __init__(self, room, video_path):
        self.room = room
        self.video_path = video_path
        self.stop_extraction_flag = False
        self.pause_extracting_flag = False
        self.reverse_frame = False
        self.forward_frame = False
        self.current_frame_index = 0
        self.frame_height = 0
        self.frame_width = 0
        self.current_frame = None
        self.current_labels = []
        self.lock = Lock()
        self.thread = None

    def stop_thread(self):
        with self.lock:
            self.stop_extraction_flag = True

    def reverse(self):
        with self.lock:
            self.reverse_frame = True
            self.forward_frame = False
            self.pause_extracting_flag = True

    def forward(self):
        with self.lock:
            self.reverse_frame = False
            self.forward_frame = True
            self.pause_extracting_flag = True

    def pause(self):
        with self.lock:
            self.pause_extracting_flag = True

    def unpause(self):
        with self.lock:
            self.pause_extracting_flag = False

    def stop(self):
        with self.lock:
            self.stop_extraction_flag = True
            self.reverse_frame = False
            self.forward_frame = False
            self.pause_extracting_flag = False

    def convert_to_base64(self, img):
        img_base64 = Image.fromarray(img)
        image_bytes = io.BytesIO()
        img_base64.save(image_bytes, format="jpeg")
        base64_string = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return base64_string

    def render(self, cap):
        ret, frame = cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = rgb_frame.copy()
        results = model(rgb_frame)
        results.render()
        self.current_labels = results.pred

        for img in results.ims:
            base64_string = self.convert_to_base64(img)
            socketio.emit("Processed_Frame", base64_string, room=self.room)

        if self.reverse_frame:
            self.reverse_frame = False

        if self.forward_frame:
            self.forward_frame = False

    def extract_frames_and_emit(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            if self.stop_extraction_flag:
                break

            if self.pause_extracting_flag:
                if self.reverse_frame:
                    self.current_frame_index = max(0, self.current_frame_index - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
                    self.render(cap)
                    continue
                elif self.forward_frame:
                    self.current_frame_index = min(total_frames - 1, self.current_frame_index + 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
                    self.render(cap)
                    continue
                else:
                    continue

            self.current_frame_index += 1
            self.render(cap)
        
        os.remove(self.video_path)
        cap.release()

    def start_extraction_thread(self):
        self.extract_frames_and_emit()

@socketio.on("Reverse")
def reverse_frame(name):
    room = name
    if users.get(room):
        users[room].reverse()
    return "Success"

@socketio.on("Forward")
def forward_frame(name):
    room = name
    if users.get(room):
        users[room].forward()
    return "Success"

@socketio.on("Pause")
def pause_session(name):
    """
    This function pauses the real time session
    """

    print(name)
    room = name
    if users.get(room):
        users[room].pause()
    return "Success"

@socketio.on("Unpause")
def pause_session(name):
    """
    This function pauses the real time session
    """
    room = name
    if users.get(room):
        users[room].unpause()
    return "Success"


@socketio.on("stop_thread")
def stop_thread(name):
    """
    This will stop the current running thread by setting the flag to True
    """
    room = name
    if users.get(room):
        users[room].stop()

    return "Success"

@socketio.on('join')
def create_new_socket(name):
    room = name
    join_room(room)
    users[room] = UserSession(room, None)
    print(f"User {room} connected.")
    return "Success"

@socketio.on('start-session')
def start_session(data):
    room = data['name']
    video_path = data['video_path']
    
    if users.get(room):
        users[room].video_path = video_path
        users[room].start_extraction_thread()
    return "Success"

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

    global FRAME_WIDTH, FRAME_HEIGHT, STOP_EXTRACTION_FLAG

    STOP_EXTRACTION_FLAG = False

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)

        FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Use CAP_PROP_FRAME_WIDTH to get the frame width
        FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        size_message = {
        "width":FRAME_WIDTH,
            "height":FRAME_HEIGHT
        }
 
        return jsonify({"ack": True, "filepath": filepath, "size":size_message}), 200

    return jsonify({'ack':False, 'error': 'Invalid file format'}), 404


@app.route('/feedback', methods=['POST'])
def get_feedback():
    """
    This function is responsible for receiving the feedback
    and saving those feedbacks in appropriate format in the server
    """
    data = request.json
    boxes = data.get('boxes')
    size = data.get('size')
    windowSize = data.get('windowSize')

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

        adjustment_factor_x = windowSize['width'] - size['width'] - 50
        adjustment_factor_y = (windowSize['height'] - size['height']) / 2 
        center_x = round((box['x'] - adjustment_factor_x + (box['width'] / 2)) / size['width'], 6)
        center_y = round((box['y'] - adjustment_factor_y + (box['height'] / 2 )) / size['height'], 6)
        center_width = round((box['width']) / size['width'], 6)
        center_height = round((box['height']) / size['height'], 6)
        
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

# @app.route('/')
# @app.route('/register-service')
# @app.route('/session')
# def index():
#     return app.send_static_file('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    socketio.run(app,
                 host='0.0.0.0', port = 8000,
                 allow_unsafe_werkzeug=True)
