from django.contrib import admin
from .models import AnimeData
from .models import *

admin.site.register(UsernameCache)
admin.site.register(TaskQueue)
admin.site.register(AnimeData)
admin.site.register(AnimeDataUpdated)

# Register your models here.
