import streamlit as st
from PIL import Image
from utils.caption_generator import generate_captions

# Streamlit UI
st.title("🖼️ Image Caption Generator")
st.write("Upload an image and generate multiple captions using BLIP!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to a PIL image
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate Captions Button
    if st.button("Generate Captions"):
        captions = generate_captions(image, num_captions=8)

        st.success("**Generated Captions:**")
        for i, caption in enumerate(captions):
            st.write(f"Caption {i+1}: {caption}")