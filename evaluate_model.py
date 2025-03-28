import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from custom_dataset import CustomDataset
from multitask_model import MultiTaskModel

# Paths and Configuration
VAL_SET_PATH = "data/val_set"
LABELS_PATH = "DCSASS_Dataset/Labels"
MODEL_PATH = "activity_detection_model.pth"
RESULTS_FILE = "evaluation_results.txt"
DATA_PATH = "DCSASS_Dataset"
ACTIVITIES = [folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder)) and folder != 'Labels']

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load validation set
num_frames = 5  # Number of frames to extract from each video
val_dataset = CustomDataset(VAL_SET_PATH, LABELS_PATH, transform, num_frames=num_frames)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model (MobileNetV2 for lightweight deployment)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(pretrained=True)

num_classes = 2
model = MultiTaskModel(model, num_frames=num_frames)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()

# Loss function
criterion_binary = nn.BCELoss().to(device)
criterion_multi = nn.NLLLoss().to(device)

# Evaluation
correct_binary = 0
total_binary = 0
correct_multi = 0
total_multi = 0
all_labels_binary = []
all_preds_binary = []
all_labels_multi = []
all_preds_multi = []
total_loss_binary = 0
total_loss_multi = 0

with torch.no_grad():
    for images, labels_binary, labels_multi in val_loader:
        images, labels_binary = images.to(device), labels_binary.to(device)
        labels_multi = labels_multi.to(device)
        outputs_binary, outputs_multi = model(images)
        
        # Calculate loss
        loss_binary = criterion_binary(outputs_binary.squeeze(), labels_binary.float())
        loss_multi = criterion_multi(outputs_multi, labels_multi)
        total_loss_binary += loss_binary.item()
        total_loss_multi += loss_multi.item()
        
        # Binary classification evaluation
        predicted_binary = (outputs_binary.squeeze() > 0.5).long()
        total_binary += labels_binary.size(0)
        correct_binary += (predicted_binary == labels_binary).sum().item()
        all_labels_binary.extend(labels_binary.cpu().numpy())
        all_preds_binary.extend(predicted_binary.cpu().numpy())

        # Multi-class classification evaluation
        _, predicted_multi = torch.max(outputs_multi.data, 1)
        total_multi += labels_multi.size(0)
        correct_multi += (predicted_multi == labels_multi).sum().item()
        all_labels_multi.extend(labels_multi.cpu().numpy())
        all_preds_multi.extend(predicted_multi.cpu().numpy())

# Calculate metrics
accuracy_binary = 100 * correct_binary / total_binary
accuracy_multi = 100 * correct_multi / total_multi
precision_binary = precision_score(all_labels_binary, all_preds_binary, average='binary')
recall_binary = recall_score(all_labels_binary, all_preds_binary, average='binary')
f1_binary = f1_score(all_labels_binary, all_preds_binary, average='binary')
precision_multi = precision_score(all_labels_multi, all_preds_multi, average='weighted')
recall_multi = recall_score(all_labels_multi, all_preds_multi, average='weighted')
f1_multi = f1_score(all_labels_multi, all_preds_multi, average='weighted')

# Save results to a .txt file
with open(RESULTS_FILE, 'w') as f:
    f.write(f"Binary Classification - Accuracy: {accuracy_binary:.2f}%, Precision: {precision_binary:.2f}, Recall: {recall_binary:.2f}, F1 Score: {f1_binary:.2f}\n")
    f.write(f"Multi-Class Classification - Accuracy: {accuracy_multi:.2f}%, Precision: {precision_multi:.2f}, Recall: {recall_multi:.2f}, F1 Score: {f1_multi:.2f}\n")

print("Evaluation complete. Results saved.")