import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.frame = img
        return img

def app():
    st.markdown(
        """
        <div class="camera-container">
            <h2>📷 Live Kamera mit Gesichtserkennung</h2>
            <p>Gesichter werden automatisch erkannt und hervorgehoben.</p>
        </div>
        """, unsafe_allow_html=True
    )

    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        if st.button("📸 Bild aufnehmen"):
            image = ctx.video_processor.frame
            if image is not None:
                st.session_state.captured_image = image
                cv2.imwrite("snapshot.jpg", image)
                st.success("Bild aufgenommen und gespeichert als snapshot.jpg.")

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_img = image[y:y+h, x:x+w]
                        resized_face = cv2.resize(face_img, (200, 200))  
                        cv2.imwrite("face.jpg", resized_face)
                        st.image(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB), caption="Gesicht", width=200)
                else:
                    st.warning("Kein Gesicht erkannt.")
            else:
                st.warning("Kein Bild verfügbar.")