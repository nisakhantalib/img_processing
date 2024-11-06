import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# Load model
model = MobileNetV2(weights='imagenet')

# Read and prepare image
img = cv2.imread('biscuit.jpg')
print("Original image size", img.shape)
# Might show something like (1000, 800, 3)
# Means: 1000 pixels height, 800 pixels width, 3 color channels (BGR)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Shape stays same, just changes BGR to RGB

img = cv2.resize(img, (224, 224))
# Means: 224x224 pixels, 3 colors (RGB) , (224,224,3)

# add batch dimension, as model expect
img_array = np.expand_dims(img, axis=0)
# Now shows (1, 224, 224, 3)

processed_img = preprocess_input(img_array)

# Predict
predictions = model.predict(processed_img)

# Show results
print("\nPredictions:")
for _, label, confidence in decode_predictions(predictions)[0]:
    print(f"{label}: {confidence*100:.2f}%")

# Show image
plt.imshow(img)
plt.title("Processed Image")
plt.show()