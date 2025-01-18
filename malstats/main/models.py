from datetime import timedelta, datetime

from django.db import models
from .model_managers import RetryManager
from django.db.utils import OperationalError
import time
import uuid

import logging

view_logger = logging.getLogger('Nyanpasutats.view')


class AnimeData(models.Model):
    mal_id = models.IntegerField()
    mean_score = models.DecimalField(max_digits=3, decimal_places=2)
    scores = models.IntegerField()
    members = models.IntegerField()
    episodes = models.PositiveSmallIntegerField(null=True)
    duration = models.DecimalField(max_digits=5, decimal_places=2, null=True)
    type = models.CharField(max_length=10, null=True)
    year = models.PositiveSmallIntegerField(null=True)
    season = models.PositiveSmallIntegerField(null=True)
    name = models.CharField(max_length=300)
    image_url = models.CharField(max_length=200, null=True)

    def __str__(self):
        return self.name

    objects = RetryManager()  # To avoid I/O conflicts from concurrent requests since we're using SQLite


class AnimeDataUpdated(models.Model):
    updated_at = models.DateTimeField(auto_now=True)
    mal_id = models.IntegerField()
    mean_score = models.DecimalField(max_digits=3, decimal_places=2)
    scores = models.IntegerField()
    members = models.IntegerField()
    episodes = models.PositiveSmallIntegerField(null=True)
    duration = models.DecimalField(max_digits=5, decimal_places=2, null=True)
    type = models.CharField(max_length=10, null=True)
    year = models.PositiveSmallIntegerField(null=True)
    season = models.PositiveSmallIntegerField(null=True)
    name = models.CharField(max_length=300)
    image_url = models.CharField(max_length=200, null=True)

    def __str__(self):
        return f"{self.name} (Updated)"

    objects = RetryManager()


class SavedImage(models.Model):
    unique_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)  # Unique identifier for the image
    user_name = models.CharField(max_length=255)  # Username
    file_name = models.CharField(max_length=255)  # Original file name
    file_path = models.FileField(upload_to='generated/')  # File path in the media directory
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp





