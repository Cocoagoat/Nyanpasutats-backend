from django.urls import path
from . import views

urlpatterns = [
    path("<int:id>", views.index, name="index"),
    path("home/", views.home, name="home"),
    path("list<int:id>/", views.list, name="list"),
    path("recs/", views.get_recommendations, name="recs"),
    path("create/", views.create, name="create"),
    path("wahaha/", views.index2, name="index2")
]