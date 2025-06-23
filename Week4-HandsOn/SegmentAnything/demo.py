from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import os

print("👉 Starting demo.py")
print("Current folder contents:", os.listdir("."))

model_type = "vit_b"
checkpoint = "sam_vit_b.pth"
print(f"Loading model type={model_type} from {checkpoint}")

sam = sam_model_registry[model_type](checkpoint=checkpoint)
print("Model loaded successfully")
sam.to(device="cpu")
print("Moved model to CPU")

mask_generator = SamAutomaticMaskGenerator(sam)
print("Mask generator created")

img_name = "test_image.jpg"
print("Looking for image file:", img_name)
if not os.path.isfile(img_name):
    print(f"❌ {img_name} not found in {os.getcwd()}")
    exit(1)

# Load and convert the image
img = Image.open(img_name).convert("RGB")
print("Image loaded:", img.size)

image_np = np.array(img)
print("Converted to NumPy array with shape:", image_np.shape)

# Generate and count masks (pass the NumPy array, not the PIL Image!)
try:
    masks = mask_generator.generate(image_np)
    print("✅ Generated", len(masks), "masks")
except Exception as e:
    print("❌ Error generating masks:", e)
    raise
