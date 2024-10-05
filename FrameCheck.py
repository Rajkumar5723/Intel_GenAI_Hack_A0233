import cv2
import streamlit as st
import numpy as np
import time

# Load the pre-trained face detection model from OpenCV (Haar Cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Face Detection App")
st.write("Position your face in front of the camera...")

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Camera not accessible")
    st.stop()

# Allow time for the user to set their face
time.sleep(5)

# Variable to track if a face has been detected
face_detected = False
face_detection_start_time = None  # To track when face detection starts
start_time = time.time()  # Start time for the 3-minute countdown

# Streamlit loop for displaying the video frames
while True:
    ret, frame = cap.read()

    if not ret:
        st.error("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        if not face_detected:
            face_detected = True
            face_detection_start_time = time.time()  # Record when the face was first detected

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        st.text("Face detected! Keep moving...")

    else:
        if face_detected:
            st.text("Face not detected! Please move back into the frame.")
            face_detected = False
        else:
            st.text("No face detected! Please move into the frame.")

    # Check if the face has been detected for 5 seconds
    if face_detected and (time.time() - face_detection_start_time >= 5):
        st.success("Face detected for 5 seconds. Terminating...")
        break

    # Display the resulting frame in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    st.image(frame, channels='RGB')

    # Check if three minutes have passed
    if time.time() - start_time > 180:  # 3 minutes = 180 seconds
        st.warning("Three minutes passed. Exiting...")
        break

# Release the capture
cap.release()
import cv2
import streamlit as st
import numpy as np
import time

# Load the pre-trained face detection model from OpenCV (Haar Cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Face Detection App")
st.write("Position your face in front of the camera...")

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Camera not accessible")
    st.stop()

# Allow time for the user to set their face
time.sleep(5)

# Variable to track if a face has been detected
face_detected = False
face_detection_start_time = None  # To track when face detection starts
start_time = time.time()  # Start time for the 3-minute countdown

# Flag to check if the first frame has been displayed
first_frame_displayed = False

# Streamlit loop for detecting the face
while True:
    ret, frame = cap.read()

    if not ret:
        st.error("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        if not face_detected:
            face_detected = True
            face_detection_start_time = time.time()  # Record when the face was first detected

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if not first_frame_displayed:
            # Display the first frame in Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            st.image(frame, channels='RGB')
            first_frame_displayed = True

        st.text("Face detected! Keep moving...")

    else:
        if face_detected:
            st.text("Face not detected! Please move back into the frame.")
            face_detected = False
        else:
            st.text("No face detected! Please move into the frame.")

    # Check if the face has been detected for 5 seconds
    if face_detected and (time.time() - face_detection_start_time >= 5):
        st.success("Face detected for 5 seconds. Terminating...")
        break

    # Check if three minutes have passed
    if time.time() - start_time > 180:  # 3 minutes = 180 seconds
        st.warning("Three minutes passed. Exiting...")
        break

# Release the capture
cap.release()
