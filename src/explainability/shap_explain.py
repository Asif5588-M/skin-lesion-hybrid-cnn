import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shap
import os

# ---- Config ----
MODEL_PATH = r"D:\Projects\skin-lesion-hybrid-cnn-workspace\skin-lesion-hybrid-cnn\models\best_model.pth"
IMAGES_DIR = r"D:\Projects\skin-lesion-hybrid-cnn-workspace\skin-lesion-hybrid-cnn\data\external\sample_images"
OUTPUT_DIR = r"D:\Projects\skin-lesion-hybrid-cnn-workspace\skin-lesion-hybrid-cnn\reports\figures"
NUM_CLASSES = 7
DEVICE = torch.device("cpu")
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

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

# ---- Image preprocessing function ----
def preprocess(img_path):
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_float = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_float - mean) / std
    return torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0).float()

# ---- Load 3 images as background + 1 test image ----
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')][:4]

background_tensors = torch.cat([preprocess(os.path.join(IMAGES_DIR, f)) for f in image_files[:3]])
test_tensor = preprocess(os.path.join(IMAGES_DIR, image_files[3]))

print(f"Background images: {image_files[:3]}")
print(f"Test image: {image_files[3]}")

# ---- SHAP GradientExplainer ----
print("Running SHAP GradientExplainer (may take a few minutes on CPU)...")
explainer = shap.GradientExplainer(model, background_tensors)
shap_values = explainer.shap_values(test_tensor)

print("SHAP values computed.")

# ---- Plot SHAP ----
test_image_display = cv2.imread(os.path.join(IMAGES_DIR, image_files[3]))
test_image_display = cv2.cvtColor(test_image_display, cv2.COLOR_BGR2RGB)
test_image_display = cv2.resize(test_image_display, (224, 224))

# Get predicted class
with torch.no_grad():
    output = model(test_tensor)
    pred_idx = output.argmax(dim=1).item()

print(f"Predicted class: {classes[pred_idx]}")

shap_img = shap_values[pred_idx][0].transpose(1, 2, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(test_image_display)
axes[0].set_title("Original Image")
axes[0].axis("off")

shap_abs = np.abs(shap_img).mean(axis=2)
axes[1].imshow(shap_abs, cmap='hot')
axes[1].set_title(f"SHAP Heatmap\nPred: {classes[pred_idx]}")
axes[1].axis("off")

axes[2].imshow(test_image_display)
axes[2].imshow(shap_abs, cmap='hot', alpha=0.5)
axes[2].set_title("SHAP Overlay")
axes[2].axis("off")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "shap_explanation.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {output_path}")