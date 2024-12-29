import streamlit as st
import cv2
import numpy as np
import pickle
from deepface import DeepFace

# Load embeddings and names
with open('known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)
known_face_encodings = np.array(known_face_encodings)

# Cosine similarity function
def cosine_similarity_vectorized(embedding, known_embeddings):
    dot_product = np.dot(known_embeddings, embedding)
    norms = np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding)
    return dot_product / norms

# Face recognition function
def recognize_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    results = []
    for (x, y, w, h) in faces:
        try:
            face_image = frame[y:y + h, x:x + w]
            face_image = cv2.resize(face_image, (224, 224))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            real_time_embedding = DeepFace.represent(img_path=face_image, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
            similarities = cosine_similarity_vectorized(real_time_embedding, known_face_encodings)
            best_match_index = np.argmax(similarities)
            best_match_similarity = similarities[best_match_index]
            
            name = known_face_names[best_match_index] if best_match_similarity > 0.5 else "Unknown"
            results.append((x, y, w, h, name))
        except Exception as e:
            print(f"Error processing face: {e}")
            results.append((x, y, w, h, "Unknown"))

    return results

# Streamlit UI
st.title("Real-Time Face Recognition")
st.text("This app performs real-time face recognition using OpenCV and DeepFace.")

# Video stream
run = st.checkbox("Run")
FRAME_WINDOW = st.image([])

video_capture = cv2.VideoCapture(0)

while run:
    ret, frame = video_capture.read()
    if not ret:
        st.write("Failed to access the webcam. Please check your camera settings.")
        break

    # Recognize faces
    results = recognize_faces(frame)

    # Draw rectangles and labels
    for (x, y, w, h, name) in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame in Streamlit
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    st.write("Stopped.")
    video_capture.release()
