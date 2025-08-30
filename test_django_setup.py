#!/usr/bin/env python3
"""
Test script to verify Django setup is working
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
    import django
    django.setup()
    
    from django.conf import settings
    from django.contrib.auth.models import User
    from detection.models import Detection
    from detection.ml_models import predictor
    
    print("✅ Django setup successful!")
    print(f"   Database: {settings.DATABASES['default']['ENGINE']}")
    print(f"   Debug mode: {settings.DEBUG}")
    print(f"   Media root: {settings.MEDIA_ROOT}")
    
    # Check if models can be imported
    print("✅ Models imported successfully!")
    
    # Check ML predictor
    print("✅ ML predictor loaded!")
    if predictor.vgg16_model is not None:
        print("   - VGG16 model: Available")
    else:
        print("   - VGG16 model: Not found (expected until trained)")
        
    if predictor.regnet_model is not None:
        print("   - RegNet model: Available")
    else:
        print("   - RegNet model: Not found (expected until trained)")
    
    print("\n🎉 Django web application is ready!")
    print("\nNext steps:")
    print("1. Run: python manage.py migrate")
    print("2. Run: python manage.py createsuperuser")
    print("3. Run: python manage.py runserver")
    print("4. Visit: http://localhost:8000")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install Django dependencies:")
    print("pip install -r django_requirements.txt")
    
except Exception as e:
    print(f"❌ Setup error: {e}")
    print("Check your Django configuration.")