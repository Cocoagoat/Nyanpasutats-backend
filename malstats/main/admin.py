from django.contrib import admin
from .models import AnimeData
from .models import *

admin.site.register(UsernameCache)
admin.site.register(TaskQueue)

# Register your models here.
