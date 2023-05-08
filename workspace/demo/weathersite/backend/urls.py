from django.urls import path

from . import views

urlpatterns = [
    path('', views.get_data, name='get_data'),
    path('hourlypredict/', views.get_hourly_predict, name='get_hourly_predict'),
    path('dailypredict/', views.get_daily_predict, name='get_daily_predict'),
]