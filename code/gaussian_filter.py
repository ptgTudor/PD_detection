import cv2
from matplotlib import pyplot as plt

# Load a colored image
image = cv2.imread("29.png")  # By default, OpenCV loads images in BGR format

# Apply Gaussian blur with different kernel sizes
blurred_image_5 = cv2.GaussianBlur(image, (3, 3), 0)
blurred_image_15 = cv2.GaussianBlur(image, (29, 29), 0)

# Convert BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
blurred_image_5_rgb = cv2.cvtColor(blurred_image_5, cv2.COLOR_BGR2RGB)
blurred_image_15_rgb = cv2.cvtColor(blurred_image_15, cv2.COLOR_BGR2RGB)

# Display the original and blurred images
plt.figure(figsize=(15, 5))
plt.imshow(image_rgb)
plt.title('Original Image')

plt.figure(figsize=(15, 5))
plt.imshow(blurred_image_5_rgb)
plt.title('Gaussian Blur (5x5)')

plt.figure(figsize=(15, 5))
plt.imshow(blurred_image_15_rgb)
plt.title('Gaussian Blur (15x15)')

plt.show()
