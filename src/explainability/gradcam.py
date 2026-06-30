import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from huggingface_hub import hf_hub_download
import os

# ---- Step 1: Model download from HuggingFace ----
print("Model download ho raha hai...")
model_path = hf_hub_download(
    repo_id="asif-nawaz-ml/skin-lesion-hybrid-resnet50-efficientnet",
    filename="best_model.pth",
    local_dir=r"D:\Projects\skin-lesion-hybrid-cnn-workspace\skin-lesion-hybrid-cnn\models"
)
print(f"Model saved: {model_path}")

# ---- Step 2: Synthetic test image banana ----
image_path = r"D:\Projects\skin-lesion-hybrid-cnn-workspace\skin-lesion-hybrid-cnn\reports\figures\sample_lesion.jpg"
os.makedirs(os.path.dirname(image_path), exist_ok=True)
sample = np.random.randint(180, 220, (224, 224, 3), dtype=np.uint8)
cv2.imwrite(image_path, sample)
print(f"Sample image created: {image_path}")

# ---- Hybrid Model class ----
NUM_CLASSES = 7
DEVICE = torch.device("cpu")

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

# ---- Step 3: Model load ----
model = HybridSkinModel(NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print("Model loaded.")

# ---- Step 4: Image preprocess ----
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (224, 224))
image_float = image_resized.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
image_normalized = (image_float - mean) / std
input_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0).float()

# ---- Step 5: Grad-CAM ----
target_layer = [model.backbone1.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layer)
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
visualization = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)

# ---- Step 6: Save + show ----
output_path = r"D:\Projects\skin-lesion-hybrid-cnn-workspace\skin-lesion-hybrid-cnn\reports\figures\gradcam_output.png"

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title("Grad-CAM Heatmap")
plt.axis("off")
plt.tight_layout()
plt.savefig(output_path)
plt.show()
print(f"Saved: {output_path}")