"""This class allows server to interact with a storage system
 by performing various operations such as reading from, writing to. """

import os
from PIL import Image
import zipfile
import subprocess
from werkzeug.utils import secure_filename

class Storage:

    @staticmethod
    def save_file(folder_dir, file):
        filename = secure_filename(file.filename)
        filepath = os.path.join(folder_dir, filename)
        file.save(filepath)
        return filepath

    @staticmethod
    def clean_system(filename):
        os.remove(os.path.join(os.getcwd(), filename))
        bash_code = """rm ./pictures/*"""
        process = subprocess.Popen(['bash', '-c', bash_code],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        
    @staticmethod
    def download_zip(room, user, folder_dir):

        pictures_folder = os.path.join(os.getcwd(), folder_dir)

        # Create a temporary directory to store the zip file
        temp_dir = os.getcwd()
        zip_filename = f'{room}_{user.diagnosis}.zip'
        zip_filepath = os.path.join(temp_dir, zip_filename)

        # Create a zip file and add all images from the pictures folder
        with zipfile.ZipFile(zip_filepath, 'w') as zip_file:
            for root, dirs, files in os.walk(pictures_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, pictures_folder)
                    zip_file.write(file_path, arcname=arcname)
        
        return zip_filepath
   
    @staticmethod
    def feedback(data, sizes, feedback_folder, current_frame, current_labels):
          
        boxes = data.get('boxes')
        size = data.get('size')
        windowSize = data.get('windowSize')

        count = int()
        label_path = str()
        count_file = os.path.join(feedback_folder, 'count_frames.txt')

        if not os.path.exists(feedback_folder):
            os.makedirs(feedback_folder)
            os.makedirs(os.path.join(feedback_folder, 'images'))
            os.makedirs(os.path.join(feedback_folder, 'labels'))
        

            with open(count_file, 'w') as file:
                count += 1
                file.write(str(count))
                file.close()

        else:
            with open(count_file, 'r') as file:
                count = int(file.read())
                count += 1
                file.close()


        rgb_image = Image.fromarray(current_frame)
        footage_name = os.path.join(feedback_folder, 'images', f"{count}.jpg")
        rgb_image.save(footage_name)

        
        for box in current_labels:
            box = box.tolist()
            print(box)
            for i in range(len(box)):
                center_x = round((box[i][0] + ((box[i][2] - box[i][0]) / 2)) / sizes["width"], 6) 
                center_y = round((box[i][1] + ((box[i][3] - box[i][1]) / 2)) / sizes["height"], 6)
                center_width = round((box[i][2] - box[i][0]) / sizes["width"], 6)
                center_height =  round((box[i][3] - box[i][1]) / sizes["height"], 6)
                label = int(box[i][5])

                labelling_data = f"{label} {center_x} {center_y} {center_width} {center_height}"
                
                label_path = os.path.join(feedback_folder, 'labels', f"{count}.txt")

                with open(label_path, 'a+') as file:
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

            label_path = os.path.join(feedback_folder, 'labels', f"{count}.txt")

            with open(label_path, 'a+') as file:
                file.write(labelling_data + '\n')
        
        for _ in range(4):
            count += 1
            replicated_footage = os.path.join(feedback_folder, 'images', f"{count}.jpg")
            replicated_label_path = os.path.join(feedback_folder, 'labels', f"{count}.txt")
            os.system(f'cp "{footage_name}" "{replicated_footage}"')
            os.system(f'cp "{label_path}" "{replicated_label_path}"')

        with open(count_file, 'w') as file:
            file.write(str(count))
            file.close()