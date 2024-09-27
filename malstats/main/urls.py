from django.urls import path, include
from . import views
urlpatterns = [
    path("recs/", views.RecommendationsView.as_view(), name="recs"),
    path("affinity/", views.AffinityFinderView.as_view(), name="affinity"),
    path("seasonal/", views.SeasonalStatsView.as_view(), name="seasonal"),
    path("img_url/", views.get_anime_img_url, name="img_url"),
    path("img_urls/", views.get_anime_img_urls, name="img_urls"),
    path('tasks/', views.get_task_data, name='tasks'),
    path('queue_pos/', views.get_queue_position, name='queue_pos'),
    path('update_img_url/', views.update_image_url_view, name='update_img_url')
]