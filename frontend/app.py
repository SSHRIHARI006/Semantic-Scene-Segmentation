import streamlit as st
import cv2
import numpy as np

from dummy_segmentation import predict_segmentation
from path_planner import compute_path
from visualization import overlay_segmentation

st.set_page_config(layout="wide")

st.title("Off-Road Autonomous Terrain Navigation")

uploaded_file = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    image = cv2.imdecode(file_bytes,1)

    col1,col2,col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image,channels="BGR")

    segmentation = predict_segmentation(image)

    overlay = overlay_segmentation(image,segmentation)

    with col2:
        st.subheader("Segmentation Map")
        st.image(overlay,channels="BGR")

    path_image = compute_path(image)

    with col3:
        st.subheader("Optimal Path")
        st.image(path_image,channels="BGR")