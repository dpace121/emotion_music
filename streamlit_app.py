import cv2
import numpy as np
import pickle as pkl
import streamlit as st

emotion_map = {0 : 'Angry',1 : 'Happy', 2 : 'Sad',3 : 'Calm'}


model = pkl.load(open('model/model_v2.pkl','rb'))
model0 = pkl.load(open('model/model_v3.pkl','rb'))


# Load the pre-trained Haar Cascade face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_from_upload(img):
    output = []
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes,1)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces and process each
    for (x, y, w, h) in faces:
        # Crop the face from the grayscale image
        face = gray_image[y:y+h, x:x+w]

        # Resize the face to 48x48 pixels
        resized_face = cv2.resize(face, (48, 48))
        output.append(resized_face)
    return output

def predict(image):
    # Normalize the image and reshape to the input shape of the model
    resized_image = image / 255.0
    resized_image = resized_image.reshape(1, 48, 48, 1)

    # Predict the emotion
    model1_pred = model.predict(resized_image)
    model1_pred[0][2],model1_pred[0][3] = model1_pred[0][3],model1_pred[0][2]
    model2_pred = model0.predict(resized_image)
    emotion_prediction = (model1_pred + model2_pred) / 2
    max_index = np.argmax(emotion_prediction)

    # Display the predicted emotion
    predicted_emotion = emotion_map[max_index]
    return predicted_emotion

def predict_faces(faces, per_row=4):
    n_faces = len(faces)
    n_rows = (n_faces + per_row - 1) // per_row  # Calculate number of rows needed

    for i in range(n_rows):
        cols = st.columns(per_row)  # Create columns for this row
        for j in range(per_row):
            index = i * per_row + j
            if index < n_faces:
                cols[j].image(faces[index], caption=predict(faces[index]), use_column_width=True)
