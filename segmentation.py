import torch
from PIL import Image
import io
from torchvision import transforms

def get_yolov5():
    model = torch.hub.load('./yolov5', 'custom',
                           path='./model/best.pt', source='local')
    model.conf = 0.1
    return model


def get_image_from_bytes(binary_image, max_size=640):
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((max_size, max_size)),
        ])

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    resized_image = transform(input_image)
    return resized_image