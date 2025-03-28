import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_frames=5):
        super(MultiTaskModel, self).__init__()
        self.num_frames = num_frames

        # Modify the base model to use 3D convolutions
        self.base_model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # Binary classifier (Suspicious or Not)
        self.binary_classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),  # Output: single probability score
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )

        # Multi-class classifier (Violence or Theft)
        self.multi_classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),  # Output: 2 classes (Violence or Theft)
            nn.LogSoftmax(dim=1)  # Ensures proper probability distribution
        )

    def forward(self, x):
        # Reshape input to (batch_size, channels, num_frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.base_model(x)
        x = torch.flatten(x, 1)

        # Pass features through both classifiers
        binary_output = self.binary_classifier(x)  # Suspicious (1) or Not (0)
        multi_output = self.multi_classifier(x)  # Violence (0) or Theft (1)

        return binary_output, multi_output