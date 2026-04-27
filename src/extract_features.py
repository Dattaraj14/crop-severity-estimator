import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.yield_labels import get_yield_loss

# Step A: Load ResNet-50
print("Loading ResNet-50 model...")
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
print("Model loaded!")

# Step B: Define how to prepare each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Step C: Function to extract features from one image
def extract_feature(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feature = model(img_tensor)
    return feature.squeeze().numpy()

# Step D: Loop through all images
data_dir = "data/raw/plantvillage dataset/color"
all_features = []
all_labels = []
all_yields = []

disease_folders = [f for f in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, f))]

print(f"Found {len(disease_folders)} disease folders")
total_images = 0

for i, disease_folder in enumerate(disease_folders):
    folder_path = os.path.join(data_dir, disease_folder)
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"[{i+1}/{len(disease_folders)}] Processing {disease_folder} ({len(images)} images)...")

    for image_file in images:
        image_path = os.path.join(folder_path, image_file)
        try:
            feat = extract_feature(image_path)
            yield_loss = get_yield_loss(disease_folder)
            all_features.append(feat)
            all_labels.append(disease_folder)
            all_yields.append(yield_loss)
            total_images += 1
        except Exception as e:
            print(f"  Skipping {image_file}: {e}")

    print(f"  Done! Total so far: {total_images} images")

# Step E: Save everything
print("Saving features to disk...")
np.save("data/features/features.npy", np.array(all_features))
np.save("data/features/labels.npy",   np.array(all_labels))
np.save("data/features/yields.npy",   np.array(all_yields))
print(f"Done! {total_images} images processed and saved.")