import streamlit as st
import cv2
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

st.set_page_config(page_title="Live Vehicle Detection", layout="wide")

st.title("🚗 Live Vehicle Detection using Hugging Face Model")
st.write("Real-time vehicle detection using YOLOv8 from Hugging Face Hub")

# ---- Load Model ----
MODEL_ID = "keremberke/yolov8n-vehicle-detection"

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    return processor, model

processor, model = load_model()

run_live = st.checkbox("Start Live Vehicle Detection")

FRAME_WINDOW = st.image([])

# ---- Webcam detection loop ----
if run_live:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam!")
            break

        # Convert image BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Process prediction
        inputs = processor(images=pil_img, return_tensors="pt")
        outputs = model(**inputs)

        result = processor.post_process_object_detection(
            outputs,
            threshold=0.4,
            target_sizes=[pil_img.size[::-1]]
        )[0]

        # Draw boxes
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = float(score)
            label_name = model.config.id2label[label.item()]
            x1, y1, x2, y2 = map(int, box.tolist())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
