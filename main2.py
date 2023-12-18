""" This is the flask application which runs the ai model for Polyps Detection"""
import io
import base64
from PIL import Image
import numpy as np
from flask import Flask, jsonify
import cv2
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
from segmentation import get_yolov8, get_yolov5

model=get_yolov5()
# model = get_yolov8()

app = Flask(__name__)
cors = CORS(app)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on('connect_with_frontend')
def handle_connect():
    """
    This code runs when a bidirectional connection is 
    established between frontend and this server.
    """
    print('Frontend connected to AI Server.')


@socketio.on("frame_incoming")
@cross_origin()
def extract_frames_from_video(frame):
    """
    The below three lines are used to decode the base64 
    encoded frame back to numpy array format which is neccesary
    for AI server to process
    """
    # pylint: disable=no-member
    frame_bytes = base64.b64decode(frame)
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    decoded_frame = cv2.imdecode(frame_np, flags=cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)
    # pylint: enable=no-member
    pil_image = Image.fromarray(rgb_frame)
    # input_image = pil_image
    input_image = np.array(rgb_frame)

    # for yolov5
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels

    # for yolov8
    # results = model(input_image, conf=0.25)

    # Process images in parallel
    for img in results.ims:
        img_base64 = Image.fromarray(img)
        image_bytes = io.BytesIO()
        img_base64.save(image_bytes, format="jpeg")
        base64_string = base64.b64encode(
            image_bytes.getvalue()).decode("utf-8")
        emit("Processed_Frame", base64_string)

    # for r in results:
    #     im_array = r.plot()  # plot a BGR numpy array of predictions
    #     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #   #  im.show()  # show image
    #   #  im.save('results.jpg')  # save image
    #     img_base64 = Image.fromarray(im)
    #     image_bytes = io.BytesIO()
    #     img_base64.save(image_bytes, format="jpeg")
    #     base64_string = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    #     emit("Processed_Frame", base64_string)

    return jsonify({"success": True})


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=8000, debug=True)
