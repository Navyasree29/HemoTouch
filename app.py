import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('model/model.h5')

# Define blood group classes
class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Streamlit web app
st.title("Blood Group Detection from Fingerprint")
st.subheader("Enter your details")

# Input fields
name = st.text_input("Name", placeholder="Enter your name")
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
age = st.slider("Age", min_value=1, max_value=120, value=25)

# Upload fingerprint image
uploaded_file = st.file_uploader("Upload Fingerprint", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fingerprint", use_container_width=True)

    # Convert image to RGB if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image to 64x64 as expected by the model
    image = image.resize((64, 64))

    # Normalize pixel values
    image_array = np.array(image) / 255.0

    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Ensure image has batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Predict the blood group
        predictions = model.predict(image_array)
        confidence = np.max(predictions)
        blood_group = class_names[np.argmax(predictions)]

        # Display the results
        st.subheader("Detection Result")
        st.write(f"**Name:** {name if name else 'N/A'}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Age:** {age}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"**Detected Blood Group:** {blood_group}")
    else:
        st.error("Uploaded image is not in the expected RGB format.")
else:
    st.info("Please upload a fingerprint image to detect the blood group.")
