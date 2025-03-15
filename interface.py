import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

# Load YOLO model
model = YOLO("model-path")  # Update with the correct model path

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Perform inference directly on NumPy array
    results = model(image_np)

    # Draw bounding boxes on the image
    for result in results:
        draw = ImageDraw.Draw(image)
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Display images
    st.image(image, caption="Processed Image with Detections", use_column_width=True)
