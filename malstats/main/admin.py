from django.contrib import admin
from .models import AnimeData
from .models import *

admin.site.register(UsernameCache)
admin.site.register(TaskQueue)
admin.site.register(AnimeData)

# Register your models here.
