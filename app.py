import streamlit as st
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image
import numpy as np

st.title("🚗 Vehicle Detection (Hugging Face Model)")

MODEL_ID = "keremberke/yolov8n-vehicle-detection"

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    return processor, model

processor, model = load_model()

img = st.camera_input("Capture vehicle image")

if img:
    image = Image.open(img)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    result = processor.post_process_object_detection(
        outputs, threshold=0.4,
        target_sizes=[image.size[::-1]]
    )[0]

    image_np = np.array(image)

    # Draw boxes manually
    import cv2

    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        name = model.config.id2label[label.item()]

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image_np, f"{name} {float(score):.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    st.image(image_np, caption="Detected Vehicles")
