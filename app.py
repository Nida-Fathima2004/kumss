import streamlit as st
import face_recognition
import numpy as np
import pickle
import cv2
import os
from PIL import Image

DATA_FILE = "face_data.pkl"


# -------------------------------------------------------
# Helper: Load stored encodings
# -------------------------------------------------------
def load_face_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {}


# -------------------------------------------------------
# Helper: Save encodings
# -------------------------------------------------------
def save_face_data(data):
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)


# -------------------------------------------------------
# Face registration
# -------------------------------------------------------
def register_face(name, image):
    np_image = np.array(image)
    rgb_image = np_image[:, :, ::-1]

    encodings = face_recognition.face_encodings(rgb_image)

    if len(encodings) == 0:
        return False, "❌ No face detected in image!"

    data = load_face_data()
    data[name] = encodings[0]
    save_face_data(data)

    return True, f"✅ Face for '{name}' has been saved!"


# -------------------------------------------------------
# Face recognition
# -------------------------------------------------------
def recognize_face(image):
    data = load_face_data()

    if not data:
        return False, "❌ No trained faces found!", None

    known_names = list(data.keys())
    known_encodings = list(data.values())

    np_image = np.array(image)
    rgb_image = np_image[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if len(face_encodings) == 0:
        return False, "❌ No face detected in test image!", None

    test_encoding = face_encodings[0]

    matches = face_recognition.compare_faces(known_encodings, test_encoding)
    distances = face_recognition.face_distance(known_encodings, test_encoding)

    name = "Unknown"
    if True in matches:
        best_index = np.argmin(distances)
        name = known_names[best_index]

    # Draw rectangle on face
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(np_image, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(np_image, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return True, name, np_image


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.title("🧠 Face Registration & Recognition App")
st.write("Upload an image to **train** a face or **detect** a face.")

tab1, tab2 = st.tabs(["📌 Register a New Face", "🔍 Recognize a Face"])


# ---------------- REGISTER TAB ------------------------
with tab1:
    st.header("Register a Face")

    name = st.text_input("Enter name of the person:")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("Register Face"):
        if name.strip() == "":
            st.error("Please enter a name.")
        elif image_file is None:
            st.error("Please upload an image.")
        else:
            image = Image.open(image_file)
            success, message = register_face(name, image)
            if success:
                st.success(message)
            else:
                st.error(message)


# ---------------- RECOGNIZE TAB ------------------------
with tab2:
    st.header("Recognize Face")

    test_image_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])

    if st.button("Detect Face"):
        if test_image_file is None:
            st.error("Please upload an image first.")
        else:
            image = Image.open(test_image_file)
            success, result, output_img = recognize_face(image)

            if not success:
                st.error(result)
            else:
                st.success(f"🎉 Detected Person: **{result}**")
                st.image(output_img, caption="Recognition Result")
