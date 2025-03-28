import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path, transform, num_frames=5):
        self.data = []
        self.labels_binary = []  # Suspicious (1) or Not (0)
        self.labels_multi = []   # Multi-class labels (Violence/Theft)
        self.transform = transform
        self.num_frames = num_frames

        # Define category mappings: Merge into Violence/Theft
        category_mappings = {
            "Assault": "Violence",
            "Fighting": "Violence",
            "Shoplifting": "Theft",
            "Stealing": "Theft"
        }

        # Define final category labels (Violence = 0, Theft = 1)
        activity_to_index = {"Violence": 0, "Theft": 1}

        print("Loading dataset with only 'Violence' and 'Theft' categories...")

        # Loop through label files
        for label_file in os.listdir(labels_path):
            if label_file.endswith(".csv"):
                activity = label_file.split('.')[0]

                # Map other activities to Violence/Theft, ignore unrelated activities
                if activity in category_mappings:
                    mapped_activity = category_mappings[activity]
                else:
                    continue  # Skip activities that are not Violence or Theft

                # Get final class index
                activity_index = activity_to_index[mapped_activity]

                # Load label CSV
                df = pd.read_csv(os.path.join(labels_path, label_file), sep=',', header=None)
                df.columns = ['filename', 'activity', 'label']

                for _, row in df.iterrows():
                    video_file = os.path.join(data_path, row['filename'] + ".mp4")
                    if os.path.exists(video_file):
                        if pd.isna(row['label']):
                            print(f"Skipping NaN label in file: {label_file}, row: {row}")
                            continue

                        # Binary label: Suspicious (1) since all remaining categories are suspicious
                        label_binary = 1  
                        label_multi = activity_index  # Multi-class label (Violence = 0, Theft = 1)

                        self.data.append(video_file)
                        self.labels_binary.append(label_binary)
                        self.labels_multi.append(label_multi)

        print(f"Final dataset size: {len(self.data)} samples (Violence & Theft only)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_file = self.data[idx]
        label_binary = self.labels_binary[idx]
        label_multi = self.labels_multi[idx]

        # Extract multiple frames from the video
        frames = self.extract_frames(video_file, self.num_frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames along the channel dimension
        frames = torch.stack(frames, dim=0)

        return frames, torch.tensor(label_binary, dtype=torch.long), torch.tensor(label_multi, dtype=torch.long)

    def extract_frames(self, video_file, num_frames):
        cap = cv2.VideoCapture(video_file)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {idx} from {video_file}")
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)

        cap.release()
        return frames