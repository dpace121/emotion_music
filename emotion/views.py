import cv2
import numpy as np
# import tensorflow as tf
from django.http import JsonResponse
import base64
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
# from .models import Song
import pickle
import ast
import pandas as pd

from emotion.models import Song
# from googleapiclient.discovery import build

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#load model
filename = "emotion/model/model_v3.pkl"
model = pickle.load(open(filename, 'rb'))

def index(request):
    # Render a template for the homepage
    return render(request, 'emotion/index.html')

# Helper function to predict emotion
def predict_emotion(image):
    img = cv2.resize(image, (48, 48))
    img = np.expand_dims(img, axis=-1)  # For grayscale images
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize to 0-1
    predictions = model.predict(img)
    print(predictions)
    emotion_map = {0 : 'Angry',1 : 'Happy', 2 : 'Sad',3 : 'Calm'}
    return emotion_map[np.argmax(predictions)]

def get_faces(img):
    output= []
     # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through the detected faces and process each
    for (x, y, w, h) in faces:
        # Crop the face from the grayscale image
        face = img[y:y+h, x:x+w]
        
        # Resize the face to 48x48 pixels
        resized_face = cv2.resize(face, (48, 48))
        output.append(resized_face)
    
    return output

@csrf_exempt
def process_emotion(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        image_data = base64.b64decode(data.split(',')[1])
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
            # Detect faces and predict mood
        faces = get_faces(image)
        if not faces:
            return JsonResponse({"error": "No faces detected. Please try again."}, status=400)
        detected_emotion = predict_emotion(faces[0])
        
        image_data_base64 = base64.b64encode(image_data).decode('utf-8')
        
    return JsonResponse({
        'emotion': detected_emotion,
        'image_data': f'data:image/jpeg;base64,{image_data_base64}'
    })

def recommend_songs(request):
    emotion = request.GET.get('emotion')  # Get the emotion parameter
    songs = Song.objects.all()  # Get all songs
    songs = Song.objects.filter(emotion=emotion, is_public=True)  # Filter songs by emotion
    return render(request, 'emotion/recommend.html', {'songs': songs, 'emotion': emotion})
