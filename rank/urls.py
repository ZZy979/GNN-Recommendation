from django.urls import path

from rank import views

app_name = 'rank'
urlpatterns = [
    path('', views.index, name='index'),
]
