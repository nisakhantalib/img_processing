import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# Load model
model = MobileNetV2(weights='imagenet')

# Read and prepare image
img = cv2.imread('badlighting2.jpeg')
print("Original image size", img.shape)
# Might show something like (1000, 800, 3)
# Means: 1000 pixels height, 800 pixels width, 3 color channels (BGR)

# Store original image before CLAHE
original_img = img.copy()
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
original_img = cv2.resize(original_img, (224, 224))
# Shape stays same, just changes BGR to RGB

# Step 1: Convert BGR to LAB
# This separates brightness (L) from color (a,b)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Step 2: Split into channels
# l = brightness channel
# a = green-red channel
# b = blue-yellow channel
l, a, b = cv2.split(lab)

# Step 3: Apply CLAHE only to the brightness channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
l_clahe = clahe.apply(l)  # Only enhance the brightness, leave colors alone

# Step 4: Merge channels back together
# We use the enhanced brightness with original colors
lab_clahe = cv2.merge((l_clahe, a, b))

# Step 5: Convert back to BGR/RGB
img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize for MobileNetV2
img = cv2.resize(img, (224, 224))
# Means: 224x224 pixels, 3 colors (RGB) , (224,224,3)

# add batch dimension for both original and CLAHE images
original_array = np.expand_dims(original_img, axis=0)
img_array = np.expand_dims(img, axis=0)
# Now shows (1, 224, 224, 3)

processed_original = preprocess_input(original_array)
processed_img = preprocess_input(img_array)

# Predict on both images
original_predictions = model.predict(processed_original)
clahe_predictions = model.predict(processed_img)

# Show results
print("\nOriginal Image Predictions:")
for _, label, confidence in decode_predictions(original_predictions)[0]:
    print(f"{label}: {confidence*100:.2f}%")

print("\nCLAHE Enhanced Image Predictions:")
for _, label, confidence in decode_predictions(clahe_predictions)[0]:
    print(f"{label}: {confidence*100:.2f}%")

# Show both images side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(img)
plt.title("CLAHE Enhanced Image")
plt.show()