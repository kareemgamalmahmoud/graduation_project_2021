from django.urls import path
from . import views

urlpatterns = [
    path('', views.test),
    path('video_feed_1/', views.video_feed_1, name="video-feed-1"),
]