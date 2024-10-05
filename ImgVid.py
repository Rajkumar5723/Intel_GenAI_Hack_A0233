import streamlit as st
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.utils import cloudinary_url

# Configure Cloudinary with your credentials
cloudinary.config(
    cloud_name='dtfszqiev',
    api_key='515185372763299',
    api_secret='HZ3CoCHlJB6XpT7RXWp8aJEqr-c'
)

def upload_file(file, resource_type):
    try:
        result = cloudinary.uploader.upload(file, resource_type=resource_type)
        return result['public_id']
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return None

def chat_input(prompt):
    return st.text_input(f"Assistant: {prompt}\nYou:", key=prompt)

def safe_int_input(prompt, min_value=None, max_value=None, default=0):
    while True:
        user_input = chat_input(prompt)
        if user_input == '':
            return default
        try:
            value = int(user_input)
            if (min_value is None or value >= min_value) and (max_value is None or value <= max_value):
                return value
            else:
                st.error(f"Please enter a value between {min_value} and {max_value}")
        except ValueError:
            st.error("Please enter a valid integer")

def safe_float_input(prompt, min_value=None, max_value=None, default=0.0):
    while True:
        user_input = chat_input(prompt)
        if user_input == '':
            return default
        try:
            value = float(user_input)
            if (min_value is None or value >= min_value) and (max_value is None or value <= max_value):
                return value
            else:
                st.error(f"Please enter a value between {min_value} and {max_value}")
        except ValueError:
            st.error("Please enter a valid number")

def edit_image(public_id):
    st.subheader("Edit Image")
    edit_option = chat_input("What would you like to do with the image? (Resize/Crop/Add text overlay/Adjust saturation/Adjust hue/Apply grayscale filter)")

    transformation = []

    if edit_option.lower() == "resize":
        width = safe_int_input("Enter new width:", min_value=1)
        height = safe_int_input("Enter new height:", min_value=1)
        transformation.append({"width": width, "height": height, "crop": "scale"})

    elif edit_option.lower() == "crop":
        width = safe_int_input("Enter crop width:", min_value=1)
        height = safe_int_input("Enter crop height:", min_value=1)
        x = safe_int_input("Enter x coordinate:", min_value=0)
        y = safe_int_input("Enter y coordinate:", min_value=0)
        transformation.append({"crop": "crop", "width": width, "height": height, "x": x, "y": y})

    elif edit_option.lower() == "add text overlay":
        text = chat_input("Enter text to overlay:")
        font_size = safe_int_input("Enter font size:", min_value=1)
        color = chat_input("Enter text color (e.g., 'red', '#FF0000'):")
        transformation.append({"overlay": {"font_family": "Arial", "font_size": font_size, "text": text, "color": color}, "gravity": "south", "y": 10})

    elif edit_option.lower() == "adjust saturation":
        saturation = safe_int_input("Enter saturation level (-100 to 100):", min_value=-100, max_value=100)
        transformation.append({"effect": f"saturation:{saturation}"})

    elif edit_option.lower() == "adjust hue":
        hue = safe_int_input("Enter hue degree (0 to 360):", min_value=0, max_value=360)
        transformation.append({"effect": f"hue:{hue}"})

    elif edit_option.lower() == "apply grayscale filter":
        transformation.append({"effect": "grayscale"})

    if chat_input("Type 'apply' to edit the image:").lower() == "apply":
        try:
            original_url, _ = cloudinary_url(public_id)
            edited_url, _ = cloudinary_url(public_id, transformation=transformation)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_url, use_column_width=True)
            with col2:
                st.subheader("Edited Image")
                st.image(edited_url, use_column_width=True)
        except Exception as e:
            st.error(f"Error editing image: {e}")

def edit_video(public_id):
    st.subheader("Edit Video")
    edit_option = chat_input("What would you like to do with the video? (Trim/Crop/Add text overlay/Adjust saturation/Adjust coolness/Adjust hue/Apply smoothing effect)")

    transformation = []

    if edit_option.lower() == "trim":
        start_time = safe_float_input("Enter start time (in seconds):", min_value=0)
        end_time = safe_float_input("Enter end time (in seconds):", min_value=start_time)
        transformation.append({"start_offset": start_time, "end_offset": end_time})

    elif edit_option.lower() == "crop":
        width = safe_int_input("Enter crop width:", min_value=1)
        height = safe_int_input("Enter crop height:", min_value=1)
        x = safe_int_input("Enter x coordinate:", min_value=0)
        y = safe_int_input("Enter y coordinate:", min_value=0)
        transformation.append({"crop": "crop", "width": width, "height": height, "x": x, "y": y})

    elif edit_option.lower() == "add text overlay":
        text = chat_input("Enter text to overlay:")
        transformation.append({"overlay": {"font_family": "Arial", "color": "white", "font_size": 60, "text": text}})

    elif edit_option.lower() == "adjust saturation":
        saturation = safe_int_input("Enter saturation level (-100 to 100):", min_value=-100, max_value=100)
        transformation.append({"effect": f"saturation:{saturation}"})

    elif edit_option.lower() == "adjust coolness":
        coolness = safe_int_input("Enter coolness level (-100 to 100):", min_value=-100, max_value=100)
        transformation.append({"effect": f"blue:{coolness}"})

    elif edit_option.lower() == "adjust hue":
        hue = safe_int_input("Enter hue degree (0 to 360):", min_value=0, max_value=360)
        transformation.append({"effect": f"hue:{hue}"})

    elif edit_option.lower() == "apply smoothing effect":
        strength = safe_int_input("Enter smoothing strength (0 to 100):", min_value=0, max_value=100)
        transformation.append({"effect": f"improve:indoor:strength:{strength}"})

    if chat_input("Type 'apply' to edit the video:").lower() == "apply":
        try:
            original_url, _ = cloudinary_url(public_id, resource_type="video")
            edited_url, _ = cloudinary_url(public_id, resource_type="video", transformation=transformation)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Video")
                st.video(original_url)
            with col2:
                st.subheader("Edited Video")
                st.video(edited_url)
        except Exception as e:
            st.error(f"Error editing video: {e}")

def main():
    st.title("Image and Video Editor")

    file_type = chat_input("Select file type (Image/Video):").lower()

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4", "mov"])

    if uploaded_file is not None:
        resource_type = "image" if file_type == "image" else "video"
        public_id = upload_file(uploaded_file, resource_type)

        if public_id:
            st.success("File uploaded successfully!")
            if file_type == "image":
                edit_image(public_id)
            else:
                edit_video(public_id)

if __name__ == "__main__":
    main()