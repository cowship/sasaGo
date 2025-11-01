from django.urls import path
from . import views

urlpatterns = [
    path('api/health/', views.api_health, name='api_health'),
    path('api/move/', views.api_move, name='api_move'),
    path('', views.index, name='index'),
]
