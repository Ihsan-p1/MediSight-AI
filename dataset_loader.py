import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

class SubsetWrapper(torch.utils.data.Dataset):
    """
    Wrapper to apply specific transforms to a Subset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_transforms(model_type, is_train=True):
    """
    Get transforms based on model type and split.
    """
    if model_type == "fatigue":
        # MobileNetV2 expects 224x224 RGB
        t_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        # Emotion/Pain: 48x48 Grayscale
        t_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            # Normalize for grayscale (approximate mean/std)
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
        
    # Add augmentation for training
    if is_train:
        if model_type == "fatigue":
            t_list.insert(0, transforms.RandomHorizontalFlip())
            t_list.insert(1, transforms.RandomRotation(10))
        else:
            t_list.insert(0, transforms.RandomHorizontalFlip())
            t_list.insert(1, transforms.RandomRotation(10))
            
    return transforms.Compose(t_list)

def get_dataloaders(model_type, data_root="data", batch_size=32, num_workers=4, pin_memory=True):
    """
    Create DataLoaders for train, val, and test.
    """
    train_transform = get_transforms(model_type, is_train=True)
    val_test_transform = get_transforms(model_type, is_train=False)

    if model_type == "emotion":
        data_dir = os.path.join(data_root, "emotion")
        # FER2013 has explicit train/test folders
        # We will split 'train' folder into train (80%) and val (20%)
        # And use 'test' folder as test set
        train_full_dir = os.path.join(data_dir, "train")
        test_dir = os.path.join(data_dir, "test")
        
        # Load datasets without transforms first (to apply later via wrapper)
        full_train_dataset = datasets.ImageFolder(train_full_dir)
        test_dataset_raw = datasets.ImageFolder(test_dir)
        
        # Split train into train/val
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
        
        # Wrap with transforms
        train_dataset = SubsetWrapper(train_subset, transform=train_transform)
        val_dataset = SubsetWrapper(val_subset, transform=val_test_transform)
        # For test_dataset_raw (ImageFolder), we can just pass transform directly if we didn't split it.
        # But for consistency let's just use ImageFolder with transform for test
        test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)
        
        classes = full_train_dataset.classes

    elif model_type in ["fatigue", "pain"]:
        if model_type == "fatigue":
            data_dir = os.path.join(data_root, "fatigue")
        else:
            data_dir = os.path.join(data_root, "pain", "processed_proxy")
            
        # Single folder structure: Split 70/15/15
        full_dataset = datasets.ImageFolder(data_dir)
        
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size])
        
        train_dataset = SubsetWrapper(train_subset, transform=train_transform)
        val_dataset = SubsetWrapper(val_subset, transform=val_test_transform)
        test_dataset = SubsetWrapper(test_subset, transform=val_test_transform)
        
        classes = full_dataset.classes
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Common DataLoader args
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": True if num_workers > 0 else False
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    
    print(f"Loaded {model_type} data: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    print(f"Classes: {classes}")
    
    return train_loader, val_loader, test_loader, len(classes)
