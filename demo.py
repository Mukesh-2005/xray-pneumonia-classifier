import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# Define model function here (copy from model.py)
def create_model(num_classes=2):
    model = models.resnet50(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    """Predict if X-ray shows pneumonia"""
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Return results
    return {
        "NORMAL": float(probabilities[0]),
        "PNEUMONIA": float(probabilities[1])
    }

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="🏥 Chest X-Ray Pneumonia Classifier",
    description="Upload a chest X-ray image to detect pneumonia. Model trained with ResNet50 transfer learning.",
    examples=[
        # Add paths to sample images if you want
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates public link