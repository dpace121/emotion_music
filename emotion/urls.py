from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process_emotion/', views.process_emotion, name='process_emotion'),
    path('recommend_songs/', views.recommend_songs, name='recommend_songs'),
]
