import unittest
import tempfile
import json
import os
from flask import Flask, request
from src.main import app, socketio  
from unittest.mock import patch

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_index_page(self):
        response = self.app.get('/')
        self.assertIn(b'<title>GastroGenius</title>', response.data)

    def test_create_new_socket(self):
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join', 'test_room')
        
        received = socketio_test_client.get_received()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['name'], 'response')
        self.assertEqual(received[0]['args'], ['Success'])

        socketio_test_client.disconnect()


    def test_websocket_connection(self):
        socketio_test_client = socketio.test_client(app)
        self.assertTrue(socketio_test_client.is_connected())
        socketio_test_client.disconnect()


    def test_extract_frames_correct_file(self):
        # Simulate joining a room before sending the video
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join', 'test_room')

        # Create a temporary video file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        # Send video file to /send-videos route
        response = self.app.post('/send-videos', data={'file': (temp_file, 'test_video.mp4'), 'name': 'test_room'})
        data = json.loads(response.data.decode('utf-8'))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['ack'])
        self.assertTrue('filepath' in data)
        self.assertTrue('size' in data)

        socketio_test_client.disconnect()
        os.remove(temp_file.name)

    def test_extract_frames_empty_file(self):
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join', 'test_room')

        # Create a temporary video file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        # Send video file to /send-videos route
        response = self.app.post('/send-videos', data={'file': (temp_file, ''), 'name': 'test_room'})
        data = json.loads(response.data.decode('utf-8'))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(data["error"],"No selected file")

        socketio_test_client.disconnect()
        os.remove(temp_file.name)

    def test_extract_frames_no_file(self):
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join', 'test_room')

        # Create a temporary video file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        # Send video file to /send-videos route
        response = self.app.post('/send-videos', data={ 'name': 'test_room'})
        data = json.loads(response.data.decode('utf-8'))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(data["error"],"No file part")

        socketio_test_client.disconnect()
        os.remove(temp_file.name)

    def test_extract_frames_invalid_file(self):
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join', 'test_room')

        # Create a temporary video file
        temp_file = tempfile.NamedTemporaryFile(suffix='.exe', delete=False)

        # Send video file to /send-videos route
        response = self.app.post('/send-videos', data={'file': (temp_file, 'temp.exe'), 'name': 'test_room'})
        data = json.loads(response.data.decode('utf-8'))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(data["error"],"Invalid file format")

        socketio_test_client.disconnect()
        os.remove(temp_file.name)

    def test_reverse_frame(self):
        # Simulate joining a room before sending the command
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join', 'test_room')

        socketio_test_client.emit('Reverse', 'test_room')
        received = socketio_test_client.get_received()
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]['name'], 'response')
        self.assertEqual(received[0]['args'], ['Success'])

        socketio_test_client.disconnect()

    def test_forward_frame(self):
        # Simulate joining a room before sending the command
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join', 'test_room')

        socketio_test_client.emit('Forward', 'test_room')
        received = socketio_test_client.get_received()
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]['name'], 'response')
        self.assertEqual(received[0]['args'], ['Success'])

        socketio_test_client.disconnect()

    def test_pause_session(self):
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join','test_room')

        socketio_test_client.emit('Pause', 'test_room')
        received = socketio_test_client.get_received()
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]['name'], 'response')
        self.assertEqual(received[0]['args'], ['Success'])

    def test_unpause_session(self):
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join','test_room')

        socketio_test_client.emit('Unpause', 'test_room')
        received = socketio_test_client.get_received()
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]['name'], 'response')
        self.assertEqual(received[0]['args'], ['Success'])

    def test_stop_thread(self):
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join','test_room')

        socketio_test_client.emit('stop_thread', 'test_room')
        received = socketio_test_client.get_received()
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]['name'], 'response')
        self.assertEqual(received[0]['args'], ['Success'])

    def test_leave_room_without_joining_one(self):
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('leave','test_room')
        received = socketio_test_client.get_received()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['name'], 'response')
        self.assertEqual(received[0]['args'], ['User not found'])
    
    def test_leave_room_after_joining(self):
        socketio_test_client = socketio.test_client(app)
        socketio_test_client.emit('join','test_room')

        socketio_test_client.emit('leave','test_room')
        received = socketio_test_client.get_received()
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]['name'], 'response')
        self.assertEqual(received[0]['args'], ['Success'])
    
if __name__ == '__main__':
    unittest.main()
