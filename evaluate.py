import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import get_data_loaders
from model import create_model
import os

def evaluate_model():
    """Evaluate the trained model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_path = os.path.expanduser("D:/chest_xray")
    _, test_loader = get_data_loaders(data_path, batch_size=16)
    
    # Load model
    model = create_model(freeze_strategy='all')
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    model.eval()
    
    # Collect predictions
    all_preds = []
    all_labels = []
    
    print("📊 Evaluating model...\n")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    print(f"✅ Test Accuracy: {accuracy:.2f}%\n")
    
    # Classification report
    class_names = ['NORMAL', 'PNEUMONIA']
    print("📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}%', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\n✅ Confusion matrix saved as 'confusion_matrix.png'")
    
    # Save results
    with open('results.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    
    print("✅ Results saved as 'results.txt'")
    
    return accuracy

if __name__ == "__main__":
    evaluate_model()