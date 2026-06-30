# Results Log

## Project 1: Hybrid Skin Lesion Classification (HAM10000)

### Baseline Model — ResNet-50 (Transfer Learning, timm)
- Date: 2026-06-30
- Dataset: HAM10000 (10,015 images, 7 classes)
- Train/Val/Test split: 70/15/15 (stratified)
- Epochs: 15
- Optimizer: Adam, lr=1e-4, ReduceLROnPlateau scheduler
- Augmentation: Albumentations (flip, rotate, brightness/contrast)
- **Final Test Accuracy: 84.36%**
- Checkpoint: saved on Google Drive (best_model.pth)
- Next step: build hybrid architecture (ResNet-50 + EfficientNet fusion) and compare against this baseline