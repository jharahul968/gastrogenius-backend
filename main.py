from fastapi import FastAPI, File, WebSocket, UploadFile, HTTPException
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
import shutil
import base64
import time

model = get_yolov5()

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
    and return image and json result""",
    version="0.0.1",
)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# function for http request to detect image and change to json 

@app.post("/object-to-json")
async def detect_polyps_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    return {"result": detect_res}

# function for http request to detect image and change to image with boxes

@app.post("/object-to-img")
async def detect_polyps_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(),
                    media_type="image/jpeg")



UPLOAD_FOLDER= 'uploads'
ALLOWED_EXTENSIONS={'mp4', 'avi', 'mkv', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# function for http request to get video from frontend and save to local

@app.post('/send-videos')
async def extract_frames(file: UploadFile = File(...)):
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")

    filename=file.filename
    filepath=os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {'ack': True, 'filepath': filepath}



async def detect_polyps_return_img(websocket: WebSocket, frame):
    input_image=frame
    
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        
        img_base64.save(bytes_io, format="jpeg")
        
        image_bytes = io.BytesIO()
        img_base64.save(image_bytes, format="PNG")

        base64_string = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        
    
    
    await websocket.send_text(base64_string)



async def extract_frames_from_video(websocket, video_path):

    cap = cv2.VideoCapture(video_path)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a PIL Image from the NumPy array
        pil_image = Image.fromarray(rgb_frame)
        
        await detect_polyps_return_img(websocket, pil_image)
        

    cap.release()
    

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    
    await websocket.accept()
    
    
    while True:
        video_path=await websocket.receive_text()
        await extract_frames_from_video(websocket, video_path)









