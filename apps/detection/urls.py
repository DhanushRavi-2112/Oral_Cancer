from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    path('upload/', views.upload_image, name='upload'),
    path('result/<uuid:detection_id>/', views.view_result, name='result'),
    path('history/', views.detection_history, name='history'),
    path('report/<uuid:detection_id>/create/', views.create_report, name='create_report'),
    path('report/<uuid:detection_id>/', views.view_report, name='view_report'),
    path('report/<uuid:detection_id>/download/', views.download_report, name='download_report'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('compare/', views.model_comparison, name='model_comparison'),
]