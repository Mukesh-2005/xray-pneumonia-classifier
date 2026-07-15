import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from dataset import get_data_loaders
from model import create_model

def train_model(num_epochs=10, batch_size=16, freeze_strategy='all'):
    """Train the X-Ray classifier"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}\n")
    
    # Load data
    data_path = os.path.expanduser("D:/chest_xray")
    train_loader, test_loader = get_data_loaders(data_path, batch_size=batch_size)
    
    # Create model
    model = create_model(freeze_strategy=freeze_strategy)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    
    # Training
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        
        # Test phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        # Print results
        print(f"\n Results:")
        print(f"   Train Loss: {avg_loss:.4f}")
        print(f"   Train Acc:  {train_acc:.2f}%")
        print(f"   Test Acc:   {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"   Saved best model! (Test Acc: {best_acc:.2f}%)")
    
    print(f"\n{'='*60}")
    print(f" Training Complete!")
    print(f"   Best Test Accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Run training
    train_model(
        num_epochs=10,        # Increase to 15-20 for better accuracy
        batch_size=16,        # Reduce to 8 or 4 if out of memory
        freeze_strategy='all' # Use 'partial' if you have good GPU
    )
