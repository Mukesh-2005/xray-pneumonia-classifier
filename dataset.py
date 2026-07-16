import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import os

def get_data_loaders(data_dir, batch_size=32, num_workers=2):
    """
    Create train and test dataloaders for chest X-ray dataset
    
    Args:
        data_dir: Path to chest_xray folder
        batch_size: Batch size (reduce if out of memory)
        num_workers: CPU workers for loading (0 if Windows issues)
    """
    
    # Transforms for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transforms for testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=test_transform
    )
    
    # Handle class imbalance with weighted sampling
    class_counts = [1341, 3875]  # NORMAL, PNEUMONIA
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

# Test the data loader
if __name__ == "__main__":
    data_path = os.path.expanduser("D:/chest_xray")
    
    print("Loading data...")
    train_loader, test_loader = get_data_loaders(data_path, batch_size=16)
    
    print(f" Train batches: {len(train_loader)}")
    print(f" Test batches: {len(test_loader)}")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f" Batch shape: {images.shape}")  # Should be [16, 3, 224, 224]
    print(f" Labels: {labels[:8]}")  # 0=NORMAL, 1=PNEUMONIA
    
    print("\n Data loader ready!")
