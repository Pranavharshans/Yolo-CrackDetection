import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("model-path")  # Update with the correct model path

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Save the uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_image_path = temp_file.name
    
    # Perform inference
    results = model(temp_image_path, save=True)
    
    # Load the processed image
    result_image = cv2.imread(temp_image_path)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(result_image, caption="Processed Image", use_column_width=True)
