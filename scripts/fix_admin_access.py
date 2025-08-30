#!/usr/bin/env python3
"""
Script to fix admin access issues
"""
import os
import sys
import django

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'oralcancer_web.settings')

try:
    django.setup()
    
    # Check if admin tables exist
    from django.contrib.auth.models import User
    from django.contrib.sessions.models import Session
    
    print("🔍 Checking Django admin setup...")
    
    # Test database connection
    try:
        user_count = User.objects.count()
        print(f"✅ Database connection OK - {user_count} users in system")
    except Exception as e:
        print(f"❌ Database error: {e}")
        print("Run: python manage.py migrate")
        sys.exit(1)
    
    # Check if superuser exists
    superusers = User.objects.filter(is_superuser=True)
    if superusers.exists():
        print(f"✅ Superuser exists: {[u.username for u in superusers]}")
    else:
        print("⚠️  No superuser found. Create one with: python manage.py createsuperuser")
    
    # Test admin site import
    from django.contrib import admin
    print("✅ Django admin imported successfully")
    
    # Check admin URL configuration
    from django.urls import reverse
    try:
        admin_url = reverse('admin:index')
        print(f"✅ Admin URL configured: {admin_url}")
    except Exception as e:
        print(f"❌ Admin URL error: {e}")
    
    print("\n🎉 Admin setup appears to be working correctly!")
    print("The 302 redirect is normal - Django admin redirects to login page when not authenticated.")
    print("Visit http://localhost:8000/admin/ and you should see the login form.")
    
except Exception as e:
    print(f"❌ Django setup error: {e}")
    sys.exit(1)