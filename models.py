import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# =============================================================
# 1. EMOTION RECOGNITION MODEL (Mini-Xception / Custom CNN)
# =============================================================
class EmotionNet(nn.Module):
    """
    A lightweight CNN for 7-class emotion recognition.
    Input: 48x48 Grayscale images.
    Output: 7 classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
    """
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Fully Connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool4(x)
        x = self.dropout3(x)
        
        # Dense
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

# =============================================================
# 2. DROWSINESS DETECTION MODEL (Transfer Learning)
# =============================================================
class DrowsinessNet(nn.Module):
    """
    Binary classifier for Drowsy vs Alert.
    Input: 224x224 RGB images.
    Output: 2 classes (Drowsy, Non Drowsy).
    Uses MobileNetV2 as a backbone for efficiency.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(DrowsinessNet, self).__init__()
        # Load MobileNetV2
        self.backbone = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
        
        # Freeze backbone (optional, can unfreeze later)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
            
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# =============================================================
# 3. PAIN DETECTION MODEL (Proxy)
# =============================================================
class PainNet(nn.Module):
    """
    Classifier for Pain Proxy (Disgust, Sadness, Surprise).
    Input: 48x48 Grayscale.
    Output: 3 classes.
    """
    def __init__(self, num_classes=3):
        super(PainNet, self).__init__()
        # Re-use a smaller version of EmotionNet architecture
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Simple test
    print("Testing Models...")
    
    # Test EmotionNet
    emo_model = EmotionNet()
    # Use batch size 2 to avoid BatchNorm error
    dummy_input = torch.randn(2, 1, 48, 48)
    out = emo_model(dummy_input)
    print(f"EmotionNet Output Shape: {out.shape} (Expected: [2, 7])")
    
    # Test DrowsinessNet
    drowsy_model = DrowsinessNet()
    dummy_rgb = torch.randn(2, 3, 224, 224)
    out = drowsy_model(dummy_rgb)
    print(f"DrowsinessNet Output Shape: {out.shape} (Expected: [2, 2])")
    
    # Test PainNet
    pain_model = PainNet()
    out = pain_model(dummy_input)
    print(f"PainNet Output Shape: {out.shape} (Expected: [2, 3])")
