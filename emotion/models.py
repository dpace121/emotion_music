from django.db import models

class Song(models.Model):
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('neutral', 'Neutral'),
    ]
    title = models.CharField(max_length=200)
    artist = models.CharField(max_length=200)
    url = models.URLField()  # YouTube link
    emotion = models.CharField(max_length=50)
    image_url = models.URLField(blank=True, null=True)  # Album art or thumbnail URL
    is_public = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.title} by {self.artist}"

