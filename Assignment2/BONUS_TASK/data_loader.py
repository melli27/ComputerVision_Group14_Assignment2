import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, test_dir, batch_size=32, num_workers=0):
    # Add random transformations to training data to artificilly increase dataset size
    # This helps prevent overfitting and makes the model more robust
    train_tfms = transforms.Compose([
        transforms.Resize((100, 100)), # Resize images to 100x100
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip
        transforms.RandomRotation(degrees=15), # Rotate randomlly
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Randomly change brightness, contrast and saturation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Randomly translate image by 10%
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  #Imagenet Normalization
    ])

    # No data augmentation for test data, only preprocessing
    test_tfms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tfms)
    
    # batch_size = number of samples used for one optimization step (power of two) maybe 32
    # num_workers = number of subprocesses to use for data loading)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, len(train_dataset.classes)
