# GenCraft

**GenCraft** is your one-stop solution for fast and efficient media editing. With a streamlined interface, you can easily crop, resize, and adjust saturation for both images and videos—all in one platform.

This platform is built using the **Cloudinary API** for efficient media processing and is implemented in **Streamlit** to provide a smooth and interactive user experience.


# Background Quality Checker

**Background Quality Checker** is an efficient tool that monitors user positioning and detects faces within the camera frame. By incorporating a time constraint, this project ensures users maintain proper positioning for optimal detection.

## Technology Stack:
- **Face Detection**: Utilizes **OpenCV** for real-time face detection and monitoring.
- **User Interface**: Built with **Streamlit** for an interactive and user-friendly experience.



# Voice Cloning with Flask and TTS

**Voice Cloning** allows you to clone a voice for a video using the user's input audio. The remaining content is generated based on this audio, ensuring a seamless and natural sound.

## How It Works:
- **Input Audio**: Users provide their audio input, which is used to clone the voice.
- **Content Generation**: Based on the cloned voice, the rest of the content is automatically filled.
- **Technologies**: Powered by Flask and TTS for efficient voice synthesis and integration.



In our **GenCraft**, sentiment analysis is primarily used to analyze user comments and likes to gauge sentiment. This powerful tool evaluates textual data, providing valuable insights into audience reactions and content effectiveness.

## Key Features:
- Analyzes user comments and likes to determine overall sentiment.
- Provides actionable insights on user engagement and content performance.
- Utilizes **spaCy** for advanced natural language processing to ensure accurate sentiment classification.
- If no comments or likes are available, the system retrieves data from previous videos or model videos to suggest content changes and enhancements.

## Technology Stack:
- **Backend**: Built with **Flask** to handle API requests and manage sentiment analysis logic.
- **Transcription**: Utilizes the **YouTube Transcript API** to extract comments and discussions from videos for analysis.

# GenCraft

## Technology Stack:
- **Frontend**: Built with **ReactJS** for a dynamic and responsive user interface.
- **Backend**: Designed to store and manage images, videos, and text efficiently using **MongoDB**.

**Reason for MongoDB**  
Its flexible schema allows for easy storage of varying media formats, ensuring scalability as your data grows.



**Pre requirements**

  Activate your Environment 
  
  pip install requirements.txt

- streamlit run FrameCheck.py

  streamlit run ImgVid.py
 
  
