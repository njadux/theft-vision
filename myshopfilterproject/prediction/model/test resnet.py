import numpy as np
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

# --- Initial Setup ---
DATASET_PATH = "./Shop DataSet"
FRAME_OUTPUT = "./frames" # Changed for local compatibility
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

# --- Main Script ---
def main():
    """
    Main function to execute the full workflow.
    """
    print("--- Video Count ---")
    for class_name in LABEL_MAP.keys():
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.exists(class_path):
            videos = [f for f in os.listdir(class_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            print(f"📁 {class_name}: {len(videos)} videos")
        else:
            print(f"Warning: Directory '{class_path}' not found.")

    print("\n--- Frame Extraction ---")
    if os.path.exists(FRAME_OUTPUT):
        shutil.rmtree(FRAME_OUTPUT)
    os.makedirs(FRAME_OUTPUT, exist_ok=True)

    for class_name, label in LABEL_MAP.items():
        category_path = os.path.join(DATASET_PATH, class_name)
        output_path = os.path.join(FRAME_OUTPUT, str(label))
        os.makedirs(output_path, exist_ok=True)
        
        if not os.path.exists(category_path):
            print(f"Warning: Directory '{category_path}' not found. Skipping.")
            continue

        for video_file in tqdm(os.listdir(category_path), desc=f"Processing {class_name}"):
            video_path = os.path.join(category_path, video_file)
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                extract_frames_from_video(video_path, output_path)

    print("Frame extraction complete!")

    print("\n--- Dataset and DataLoader Setup ---")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = VideoFrameDataset(FRAME_OUTPUT, transform=transform)
    
    if len(full_dataset) == 0:
        print("No frames found. Exiting.")
        return

    # Use a more standard and robust way to create a stratified split
    # Get indices for stratification
    indices = np.arange(len(full_dataset))
    labels = np.array([full_dataset[i][1] for i in indices])
    
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Use `num_workers=0` as a temporary fix on Windows if the issue persists
    # The ideal fix is to put classes at the top-level, which we have done.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    print("\n--- Model Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_MAP))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0

    for epoch in range(5):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        # tqdm wrapper needs to be around the inner loop, not the outer loop
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if imgs is None or labels is None:
                continue
            
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Train Loss: {running_loss:.4f}, Acc: {train_acc:.2f}%")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                if imgs is None or labels is None:
                    continue
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "best_model.pth"))
            print("✅ Best model saved!")

        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print("-" * 50) 
        
    torch.save(model.state_dict(), "TEST_resnet_model.pth")
    print("\nTraining complete. Final model saved.")

if __name__ == "__main__":
    # Add this check to prevent issues with multiprocessing on Windows
    # It ensures that all code runs only in the main process
    try:
        main()
    except RuntimeError as e:
        if "An attempt has been made to start a new process before the current process has finished its bootstrapping phase" in str(e):
            print("Restarting with `if __name__ == '__main__'` block for multiprocessing safety.")
            print("Please ensure your entire script is wrapped within this block or run your code in a script file.")
        else:
            raise