import io
import os
import base64
from threading import Lock
from segmentation import get_yolov5
import cv2
import json
import random
from PIL import Image
import zipfile
from multiprocessing import Process
from threading import Thread

class Server:
    
    ALLOWED_EXTENSIONS = {'avi', 'mp4', 'mov', 'mkv'}
    UPLOAD_FOLDER = 'uploads'
    FEEDBACK_FOLDER = 'feedback'
    FOOTAGE_FOLDER = 'pictures'
    COUNT_FILE = "count_frames.txt"
    MODEL = get_yolov5()

    def __init__(self, room, video_path, socketio):
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
        self.detected_frames = []
        self.thread = None
        self.lock = Lock()
        self.diagnosis = None
        self.socket = socketio
        self.save_picture = False


    @staticmethod
    def allowed_file(filename):
        """
        This function defines the type of files allowed in the extraction
        """
        return '.' in filename and filename.split('.', 1)[1].lower() in Server.ALLOWED_EXTENSIONS
    

    @classmethod
    def feedback():
        pass


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
        results = Server.MODEL(rgb_frame)
        results.render()
                
        self.current_labels = results.pred

        if self.save_picture and (len( self.current_labels[0].tolist())): # the self.current_labels is tensor which has data in format tensor([[x_center y_center width_center height _center]])
            save_path = os.path.join(os.getcwd(), Server.FOOTAGE_FOLDER,f"{random.randint(1,100)}.jpg")
            image = Image.fromarray(results.ims[0])
            image.save(save_path)

        for img in results.ims:
            base64_string = self.convert_to_base64(img)
            self.socket.emit("Processed_Frame", base64_string, room=self.room)
            
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
            # break
         
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        if self.save_picture:
            data = {
                "name":self.room,
                "diagnosis":self.diagnosis
            }
            self.socket.emit("end", data, room=self.room)
        cap.release()


    def start_extraction_thread(self):
       process = Process(target= self.extract_frames_and_emit())
       process.start()
        #self.extract_frames_and_emit()
