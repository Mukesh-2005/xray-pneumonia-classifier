import torch
import torch.nn as nn
import torchvision.models as models

def create_model(freeze_strategy='all', num_classes=2):
    """
    Create X-Ray classifier using ResNet50 transfer learning
    
    Args:
        freeze_strategy: 'all' (Pattern 1) or 'partial' (Pattern 2)
        num_classes: 2 (NORMAL, PNEUMONIA)
    """
    
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    if freeze_strategy == 'all':
        # Pattern 1: Freeze everything
        for param in model.parameters():
            param.requires_grad = False
        print(" Pattern 1: All layers frozen")
        
    elif freeze_strategy == 'partial':
        # Pattern 2: Freeze only early layers
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        print(" Pattern 2: Early layers frozen")
    
    # Replace final layer
    num_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(num_features, num_classes)
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f" Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
    
    return model

# Test the model
if __name__ == "__main__":
    print("Creating model with Pattern 1...")
    model = create_model(freeze_strategy='all')
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    output = model(dummy_input)
    
    print(f"\n Input shape: {dummy_input.shape}")
    print(f" Output shape: {output.shape}")  # Should be [2, 2]
    print(f" Output: {output}")
    
    # Check if GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✅ Device: {device}")
    
    print("\n Model ready!")
