from django.db import models


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


class TaskQueue(models.Model):
    task_id = models.CharField(max_length=50, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='pending')

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return self.task_id


class UsernameCache(models.Model):
    username = models.CharField(max_length=30, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)



