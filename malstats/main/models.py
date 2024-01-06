from django.db import models


class AnimeData(models.Model):
    image_url = models.CharField(max_length=200)


