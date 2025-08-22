from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import shutil
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

LABEL_MAP = {
    "non shop lifters": 0,
    "shop lifters": 1
}

# --- Video Frame Extraction Function ---
def extract_frames_from_video(video_path, save_dir, max_frames=5):
    """
    Extracts a specified number of frames from a video and saves them.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Warning: Could not read video file at {video_path} or it has no frames.")
        cap.release()
        return

    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        
        if success:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_name = f"{video_name}_frame{i}.jpg"
            cv2.imwrite(os.path.join(save_dir, frame_name), frame)
    
    cap.release()

# --- Custom PyTorch Dataset ---
# This class needs to be at the top level
class VideoFrameDataset(Dataset):
    """
    A custom PyTorch Dataset for loading video frames organized by class labels.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory '{root_dir}' not found.")

        print("Creating dataset index...")
        for label_dir in os.listdir(root_dir):
            if not label_dir.isdigit():
                continue

            class_label = int(label_dir)
            class_path = os.path.join(root_dir, label_dir)

            if os.path.isdir(class_path):
                self.samples.extend([
                    (os.path.join(class_path, frame_name), class_label)
                    for frame_name in os.listdir(class_path)
                    if not frame_name.startswith('.')
                ])

        print(f"Found {len(self.samples)} image files across all classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)
            
        return image, label

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])