import streamlit as st
import cv2
import time
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image
import numpy as np


st.set_page_config(page_title="Live Vehicle Detection", layout="wide")
st.title("🚗 Live Vehicle Detection (Hugging Face Model)")

MODEL_ID = "keremberke/yolov8n-vehicle-detection"

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    return processor, model

processor, model = load_model()

start = st.button("Start Webcam")
FRAME_WINDOW = st.image([])

if start:
    cap = cv2.VideoCapture(0)
    st.write("Webcam started. Stop the app to restart.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read from webcam.")
            st.stop()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        inputs = processor(images=pil, return_tensors="pt")
        outputs = model(**inputs)

        result = processor.post_process_object_detection(
            outputs,
            threshold=0.4,
            target_sizes=[pil.size[::-1]]
        )[0]

        # Draw boxes
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            x1, y1, x2, y2 = map(int, box.tolist())
            name = model.config.id2label[label.item()]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{name} {float(score):.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")
        time.sleep(0.02)
