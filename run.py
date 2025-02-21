
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("model-path")

# Path to test image
test_image_path = "image-path"  # Change this to your image path

# Perform inference
results = model(test_image_path, save=True)

# Show image with detections
img = cv2.imread(test_image_path)
cv2.imshow("YOLO Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
