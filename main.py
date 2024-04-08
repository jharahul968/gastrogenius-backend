""" This is the flask application which 
    runs the ai model for Polyps Detection"""
import os
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room
from Class.server import Server
from Class.storage import Storage

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
    Storage.clean_system(filename)
    socketio.emit('response', 'Success')


@app.route('/download-zip', methods=["POST"])
def download_zip():
    """This function is responsible for downloading the zip file"""
    data = request.json
    room = data.get('name')
    zip_filepath = Storage.download_zip(room, users[room], app.config["FOOTAGE_FOLDER"])
    return send_file(zip_filepath, as_attachment=True), 200


@app.route('/feedback', methods=['POST'])
def get_feedback():
    """
    This function is responsible for receiving the feedback
    and saving those feedbacks in appropriate format in the server
    """
    data = request.json
    room = data.get('name')

    sizes = {
        "height":users[room].frame_height,
        "width":users[room].frame_width
    }

    Storage.feedback(data, sizes, app.config['FEEDBACK_FOLDER'], 
                     users[room].current_frame, users[room].current_labels)

    return jsonify({'message':'successful'})


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
        filepath = Storage.save_file(app.config["UPLOAD_FOLDER"], file)

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
