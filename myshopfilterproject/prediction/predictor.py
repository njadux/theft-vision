# myshoplifterproject/prediction/predictor.py
# THE LOGIC TO LOAD THE MODEL

import torch
import torch.nn as nn
from torchvision.models import resnet18
from .model_utils import transform, extract_frames_from_video, LABEL_MAP
import os
from PIL import Image
import shutil

_model = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    global _model
    if _model is None:
        _model = resnet18(weights=None)
        _model.fc = nn.Linear(_model.fc.in_features, len(LABEL_MAP))
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'TEST_resnet_model.pth')
        _model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        _model.eval()
        _model = _model.to(DEVICE)
    return _model

def predict_video(video_path):
    model = get_model()
    if model is None:
        return "Error: Model not loaded."

    temp_frame_dir = "./temp_frames"
    os.makedirs(temp_frame_dir, exist_ok=True)
    
    extract_frames_from_video(video_path, temp_frame_dir, max_frames=1)
    
    frame_files = [f for f in os.listdir(temp_frame_dir) if f.lower().endswith(('.jpg'))]
    if not frame_files:
        return "No frames extracted."

    frame_path = os.path.join(temp_frame_dir, frame_files[0])
    image = Image.open(frame_path).convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class_index = torch.max(output, 1)

    predicted_label = "theft detected" if predicted_class_index.item() == 1 else "safe no thefts"
     
    shutil.rmtree(temp_frame_dir)
    
    return predicted_label