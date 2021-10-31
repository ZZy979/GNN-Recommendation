from django.urls import path

from . import views

app_name = 'rank'
urlpatterns = [
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.RegisterView.as_view(), name='register'),

    path('', views.index, name='index'),
    path('search-paper/', views.SearchPaperView.as_view(), name='search-paper'),
    path('paper/<int:pk>/', views.PaperDetailView.as_view(), name='paper-detail'),
    path('author/<int:pk>/', views.AuthorDetailView.as_view(), name='author-detail'),
    path('author-rank/', views.AuthorRankView.as_view(), name='author-rank'),
    path('search-author/', views.SearchAuthorView.as_view(), name='search-author'),
]
