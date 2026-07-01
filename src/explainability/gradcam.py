import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# ---- Config ----
MODEL_PATH = r"D:\Projects\skin-lesion-hybrid-cnn-workspace\skin-lesion-hybrid-cnn\models\best_model.pth"
IMAGES_DIR = r"D:\Projects\skin-lesion-hybrid-cnn-workspace\skin-lesion-hybrid-cnn\data\external\sample_images"
OUTPUT_DIR = r"D:\Projects\skin-lesion-hybrid-cnn-workspace\skin-lesion-hybrid-cnn\reports\figures"
NUM_CLASSES = 7
DEVICE = torch.device("cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Hybrid Model ----
class HybridSkinModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone1 = timm.create_model("resnet50", pretrained=False, num_classes=0)
        feat1_dim = self.backbone1.num_features
        self.backbone2 = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        feat2_dim = self.backbone2.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feat1_dim + feat2_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f1 = self.backbone1(x)
        f2 = self.backbone2(x)
        return self.classifier(torch.cat([f1, f2], dim=1))

# ---- Load Model ----
model = HybridSkinModel(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

# ---- Class labels ----
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# ---- Grad-CAM on all 10 images ----
target_layer = [model.backbone1.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layer)

image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
print(f"Processing {len(image_files)} images...")

fig, axes = plt.subplots(len(image_files), 2, figsize=(10, len(image_files) * 4))

for idx, img_file in enumerate(image_files):
    # Load and preprocess
    img_path = os.path.join(IMAGES_DIR, img_file)
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_float = image_resized.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_float - mean) / std
    input_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0).float()

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

    # Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    visualization = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)

    # Plot
    axes[idx, 0].imshow(image_rgb)
    axes[idx, 0].set_title(f"Original: {img_file}", fontsize=8)
    axes[idx, 0].axis("off")

    axes[idx, 1].imshow(visualization)
    axes[idx, 1].set_title(f"Grad-CAM | Pred: {classes[pred_idx]} ({confidence:.2f})", fontsize=8)
    axes[idx, 1].axis("off")

    print(f"{img_file} -> Predicted: {classes[pred_idx]} (confidence: {confidence:.2f})")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "gradcam_real_images.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {output_path}")