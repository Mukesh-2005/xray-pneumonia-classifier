# 🏥 Chest X-Ray Pneumonia Classifier

Deep learning model to detect pneumonia from chest X-ray images using transfer learning with ResNet50.

## 🎯 Results

- **Test Accuracy:** 90.06%
- **Precision (Pneumonia):** 91%
- **Recall (Pneumonia):** 94%

## 🔧 Tech Stack

- **Framework:** PyTorch
- **Model:** ResNet50 (ImageNet pre-trained)
- **Transfer Learning:** Pattern 1 (Feature Extraction)
- **Dataset:** 5,863 chest X-ray images
- **Interface:** Gradio

## 📊 Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| NORMAL | 0.89 | 0.84 | 0.86 |
| PNEUMONIA | 0.91 | 0.94 | 0.92 |

## 🚀 Try It Live

[Demo on Hugging Face Spaces](https://huggingface.co/spaces/StarMukesh/xray-pneumonia-classifer)

## 💻 Run Locally

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/xray-classifier.git
cd xray-classifier

# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py
```

## 📁 Project Structure
├── train.py           # Training script
├── model.py           # Model definition
├── dataset.py         # Data loading
├── evaluate.py        # Evaluation metrics
├── demo.py            # Gradio interface
├── best_model.pth     # Trained weights
└── README.md

## 📚 Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Training: 5,216 images
- Testing: 624 images
- Classes: NORMAL vs PNEUMONIA

## 🧠 Model Architecture

- Base: ResNet50 (pretrained on ImageNet)
- Frozen: All layers except final FC
- Trainable params: 4,098 (0.02%)
- Training time: ~40 min (CPU)

## 📈 Training Details

- Epochs: 10
- Batch size: 16
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Data augmentation: Random horizontal flip

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first.

## 📝 License

MIT

## 👤 Author

**Mukesh K**
- GitHub: [@Mukesh-2005](https://github.com/Mukesh-2005)
- LinkedIn: [Mukesh K](https://linkedin.com/in/mukesh-k)

## 🙏 Acknowledgments

- Dataset by Paul Mooney (Kaggle)
- PyTorch & torchvision teams
- ResNet architecture by Microsoft Research
