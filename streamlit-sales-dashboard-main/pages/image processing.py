import streamlit as st
from PIL import Image
import numpy as np

# Function to perform image processing techniques
def apply_image_processing(image, techniques):
    processed_image = image.copy()
    if "Resize" in techniques:
        # Resize the image to a specified size (e.g., 100x100)
        processed_image = processed_image.resize((100, 100))
    if "Grayscale Conversion" in techniques:
        # Convert the image to grayscale
        processed_image = processed_image.convert("L")
    if "Image Cropping" in techniques:
        # Crop the image to a specified region (e.g., top-left corner)
        processed_image = processed_image.crop((0, 0, 50, 50))
    if "Image Rotation" in techniques:
        # Rotate the image by a specified angle (e.g., 45 degrees)
        processed_image = processed_image.rotate(45)
    return processed_image

# Main Streamlit app
def main():
    st.title("Image Processing App")
    
    # Upload images
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    
    if uploaded_files:
        # Display uploaded images
        st.subheader("Uploaded Images")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
            
            # Image processing techniques selection
            st.subheader("Image Processing Techniques")
            techniques = st.multiselect("Select Techniques to Apply", ["Resize", "Grayscale Conversion", "Image Cropping", "Image Rotation"])
            
            # Apply selected techniques
            if st.button("Apply Techniques"):
                processed_image = apply_image_processing(image, techniques)
                st.subheader("Processed Image")
                st.image(processed_image, use_column_width=True, caption="Processed Image")
            
            st.markdown("---")

if __name__ == "__main__":
    main()
