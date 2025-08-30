"""
URL configuration for oralcancer_web project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Authentication
    path('accounts/', include('apps.accounts.urls')),
    
    # Apps
    path('', include('apps.pages.urls')),
    path('detection/', include('apps.detection.urls')),
    path('dashboard/', include('apps.dashboard.urls')),
    
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# Custom error pages
handler404 = 'apps.pages.views.error_404'
handler500 = 'apps.pages.views.error_500'

# Admin site customization
admin.site.site_header = "OralScan AI Administration"
admin.site.site_title = "OralScan AI Admin"
admin.site.index_title = "Welcome to OralScan AI Administration"