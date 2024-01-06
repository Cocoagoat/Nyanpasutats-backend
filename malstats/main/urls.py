from django.urls import path, include
from . import views
urlpatterns = [
    path('', include('frontend.urls')),
    # path("<int:id>", views.index, name="index"),
    # path("home/", views.home, name="home"),
    # path("list<int:id>/", views.list, name="list"),
    # path("recs/", views.get_recommendations, name="recs"),
    # path("create/", views.create, name="create"),
    # path("wahaha/", views.index2, name="index2"),
    path("recs/", views.RecommendationsView.as_view(), name="recs"),
    path("affinity/", views.RecommendationsView.as_view(), name="affinity"),
    path("seasonal/", views.SeasonalStatsView.as_view(), name="seasonal")
]