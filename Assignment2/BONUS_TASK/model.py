import torch
import torch.nn as nn
from torchvision import models

def get_scene_model(num_classes=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load a pre-trained ResNet18 model from pytorch
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    #  Replace last fully connected layers (10 Classes) 
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Freeze backbone, unfreeze only classifier
    for param in model.parameters():
        param.requires_grad = False
        
    # Only make Classifier trainable    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model = model.to(device)
    return model