import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from custom_dataset import CustomDataset
from multitask_model import MultiTaskModel

# Paths
TRAIN_SET_PATH = "data/train_set"
VAL_SET_PATH = "data/val_set"
MODEL_PATH = "activity_detection_model.pth"
LOG_FILE = "training_log.txt"
DATA_PATH = "DCSASS_Dataset"
LABELS_PATH = "DCSASS_Dataset/Labels"
ACTIVITIES = [folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder)) and folder != 'Labels' and folder != '.ipynb_checkpoints']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
num_frames = 5  # Number of frames to extract from each video
train_dataset = CustomDataset(TRAIN_SET_PATH, LABELS_PATH, transform, num_frames=num_frames)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model (MobileNetV2 for lightweight deployment)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(pretrained=True)

num_classes = 2
print(f"Number of classes: {num_classes}")
model = MultiTaskModel(model, num_frames=num_frames)
model = model.to(device)

# Loss & Optimizer
criterion_binary = nn.BCELoss().to(device)
criterion_multi = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
with open(LOG_FILE, 'w') as f:
    for epoch in range(num_epochs):
        model.train()
        total_loss_binary = 0
        total_loss_multi = 0

        for images, labels_binary, labels_multi in train_loader:
            images, labels_binary = images.to(device), labels_binary.to(device)
            labels_multi = labels_multi.to(device)

            optimizer.zero_grad()
            outputs_binary, outputs_multi = model(images)
                       
            loss_binary = criterion_binary(outputs_binary.squeeze(), labels_binary.float())
            loss_multi = criterion_multi(outputs_multi, labels_multi)
            loss = loss_binary + loss_multi
            loss.backward()
            optimizer.step()

            total_loss_binary += loss_binary.item()
            total_loss_multi += loss_multi.item()

        epoch_loss_binary = total_loss_binary / len(train_loader)
        epoch_loss_multi = total_loss_multi / len(train_loader)
        log_msg = f"Epoch [{epoch+1}/{num_epochs}], Binary Loss: {epoch_loss_binary}, Multi-class Loss: {epoch_loss_multi}\n"
        print(log_msg)
        f.write(log_msg)

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print("Training complete. Model saved.")