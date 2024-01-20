""" This is the flask application which 
    runs the ai model for Polyps Detection"""
import os
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import zipfile
import subprocess
from flask_socketio import SocketIO, join_room, leave_room
from ServerClass.server import Server

users = {}
app = Flask(__name__, static_folder = './build', static_url_path = '/')
app.secret_key = '__your_secret_key_-'
CORS(app, origins="*")
app.config['UPLOAD_FOLDER'] = Server.UPLOAD_FOLDER
app.config['FEEDBACK_FOLDER'] = Server.FEEDBACK_FOLDER
app.config['FOOTAGE_FOLDER'] = Server.FOOTAGE_FOLDER
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("Reverse")
def reverse_frame(name):
    room = name
    if users.get(room):
        users[room].reverse()
    socketio.emit('response', 'Success')


@socketio.on("Forward")
def forward_frame(name):
    room = name
    if users.get(room):
        users[room].forward()
    socketio.emit('response', 'Success')


@socketio.on("Pause")
def pause_session(name):
    """
    This function pauses the real time session
    """
    room = name
    if users.get(room):
        users[room].pause()
    socketio.emit('response', 'Success')


@socketio.on("Unpause")
def unpause_session(name):
    """
    This function pauses the real time session
    """
    room = name
    if users.get(room):
        users[room].unpause()
    socketio.emit('response', 'Success')

@socketio.on("stop_thread")
def stop_thread(name):
    """
    This will stop the current running thread by 
    setting the flag to True
    """
    room = name
    if users.get(room):
        users[room].stop()
    socketio.emit('response', 'Success')


@socketio.on('join')
def create_new_socket(name):
    room = name
    join_room(room)
    users[room] = Server(room, None, socketio)
    socketio.emit('response', 'Success')


@socketio.on('leave')
def leave(name):
    """User will leave the room"""
    room = name
    if name not in users.keys():
       socketio.emit('response', 'User not found')
       return
    
    leave_room(room)
    del users[room]
    socketio.emit('response', 'Success')

@socketio.on('start-session')
def start_session(data):

    room = data['name']
    diagnosis = data['diagnosis']
    is_save = data['save_value']
    video_path = data['video_path']
    if users.get(room):
        users[room].video_path = video_path
        users[room].diagnosis = diagnosis
        users[room].save_picture = is_save
        users[room].start_extraction_thread()
 
    socketio.emit('response', 'Success')

@socketio.on('clean')
def clean_session(filename):
    os.remove(os.path.join(os.getcwd(), filename))
    bash_code = """rm ./pictures/*"""
    process = subprocess.Popen(['bash', '-c', bash_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    socketio.emit('response', 'Success')

@app.route('/download-zip', methods=["POST"])
def download_zip():

    data = request.json
    room = data.get('name')
    pictures_folder = os.path.join(os.getcwd(), app.config["FOOTAGE_FOLDER"])

    # Create a temporary directory to store the zip file
    temp_dir = os.getcwd()
    zip_filename = f'{room}_{users[room].diagnosis}.zip'
    zip_filepath = os.path.join(temp_dir, zip_filename)

    # Create a zip file and add all images from the pictures folder
    with zipfile.ZipFile(zip_filepath, 'w') as zip_file:
        for root, dirs, files in os.walk(pictures_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, pictures_folder)
                zip_file.write(file_path, arcname=arcname)

    # Send the zip file to the user for download
    return send_file(zip_filepath, as_attachment=True), 200


@app.route('/send-videos', methods=["POST"])
def extract_frames():
    """
    This function handles the incoming video file from the
    frontend and then saves it in uploads folder. It then returns
    an acknowledgement and the video path if the operation is successful.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 404

    file = request.files['file']
    room = request.form['name']
    users[room].stop_extraction_flag = False

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 404

    if file and Server.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        #pylint: disable=no-member
        cap = cv2.VideoCapture(filepath)
        users[room].frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        users[room].frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #pylint: enable=no-member

        size_message = {
        "width":users[room].frame_width,
            "height":users[room].frame_height
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
    room = data.get('name')
    windowSize = data.get('windowSize')

    count = int()
    count_file = os.path.join(app.config['FEEDBACK_FOLDER'], 'count_frames.txt')

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
    
    rgb_image = Image.fromarray(users[room].current_frame)
    footage_name = os.path.join(app.config['FEEDBACK_FOLDER'], 'images', f"{count}.jpg")
    rgb_image.save(footage_name)

    
    for box in users[room].current_labels:
        box = box.tolist()
        for i in range(len(box)):
            center_x = round((box[i][0] + ((box[i][2] - box[i][0]) / 2)) / users[room].frame_height, 6) 
            center_y = round((box[i][1] + ((box[i][3] - box[i][1]) / 2)) / users[room].frame_height, 6)
            center_width = round((box[i][2] - box[i][0]) / users[room].frame_width, 6)
            center_height =  round((box[i][3] - box[i][1]) / users[room].frame_height, 6)
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


@app.route('/')
@app.route('/register-service')
@app.route('/session')
def index():
    return app.send_static_file('index.html') 


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if not os.path.exists(app.config['FOOTAGE_FOLDER']):
        os.makedirs(app.config['FOOTAGE_FOLDER'])

    socketio.run(app,debug=True,
                 host='0.0.0.0', port = 8000,
                 allow_unsafe_werkzeug=True)
