import cv2
import numpy as np
import tensorflow as tf
from django.http import JsonResponse
import base64
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .models import Song

# Load the model
model = tf.keras.models.load_model('emotion/model/model.h5')

def index(request):
    # Render a template for the homepage
    return render(request, 'emotion/index.html')

# Helper function to predict emotion
def predict_emotion(image):
    img = cv2.resize(image, (48, 48))
    img = np.expand_dims(img, axis=0) / 255.0
    predictions = model.predict(img)
    emotion_labels = ["Happy", "Sad", "Neutral"]
    return emotion_labels[np.argmax(predictions)]

@csrf_exempt
def process_emotion(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        image_data = base64.b64decode(data.split(',')[1])
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

        detected_emotion = predict_emotion(image)
        return JsonResponse({
            'emotion': detected_emotion,
             'image_data': f'data:image/jpeg;base64,{image_data}'
            })
    
def recommend_songs(request):
    emotion = request.GET.get('emotion')  # Get the emotion parameter
    songs = Song.objects.filter(emotion=emotion)  # Filter songs by emotion
    return render(request, 'emotion/recommend.html', {'songs': songs, 'emotion': emotion})
