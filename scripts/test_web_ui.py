#!/usr/bin/env python
"""
Test all web UI pages and fix any issues
"""
import os
import sys
import django

# Setup Django
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.oralcancer_web.settings')
django.setup()

from django.urls import reverse, resolve
from django.test import Client
from django.contrib.auth.models import User

def test_all_urls():
    """Test all URLs in the application"""
    client = Client()
    
    # Create test user
    try:
        test_user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        print("✓ Test user created")
    except:
        test_user = User.objects.get(username='testuser')
        print("✓ Test user already exists")
    
    # Public URLs (no authentication required)
    public_urls = [
        ('/', 'Home Page'),
        ('/accounts/login/', 'Login Page'),
        ('/accounts/signup/', 'Signup Page'),
    ]
    
    # Authenticated URLs
    auth_urls = [
        ('/dashboard/', 'Dashboard'),
        ('/detection/upload/', 'Upload Page'),
        ('/detection/history/', 'History Page'),
        ('/accounts/profile/', 'Profile Page'),
    ]
    
    print("\n" + "="*50)
    print("Testing Public URLs")
    print("="*50)
    
    for url, name in public_urls:
        try:
            response = client.get(url)
            status = "✓" if response.status_code in [200, 302] else "✗"
            print(f"{status} {name}: {url} - Status: {response.status_code}")
            
            if response.status_code == 404:
                print(f"  → URL pattern not found")
            elif response.status_code == 500:
                print(f"  → Server error - check views and templates")
        except Exception as e:
            print(f"✗ {name}: {url} - Error: {str(e)}")
    
    print("\n" + "="*50)
    print("Testing Authenticated URLs")
    print("="*50)
    
    # Login
    client.login(username='testuser', password='testpass123')
    
    for url, name in auth_urls:
        try:
            response = client.get(url)
            status = "✓" if response.status_code in [200, 302] else "✗"
            print(f"{status} {name}: {url} - Status: {response.status_code}")
            
            if response.status_code == 404:
                print(f"  → URL pattern not found")
            elif response.status_code == 500:
                print(f"  → Server error - check views and templates")
        except Exception as e:
            print(f"✗ {name}: {url} - Error: {str(e)}")
    
    # Test static files
    print("\n" + "="*50)
    print("Testing Static Files")
    print("="*50)
    
    static_files = [
        ('/static/css/style.css', 'Main CSS'),
        ('/static/images/hero-medical.svg', 'Hero Image'),
    ]
    
    for url, name in static_files:
        try:
            response = client.get(url)
            status = "✓" if response.status_code == 200 else "✗"
            print(f"{status} {name}: {url} - Status: {response.status_code}")
        except Exception as e:
            print(f"✗ {name}: {url} - Error: {str(e)}")

def check_templates():
    """Check if all templates exist"""
    print("\n" + "="*50)
    print("Checking Templates")
    print("="*50)
    
    templates_dir = os.path.join(project_root, 'templates')
    
    required_templates = [
        'base.html',
        'pages/home.html',
        'pages/about.html',
        'pages/contact.html',
        'registration/login.html',
        'registration/signup.html',
        'dashboard/home.html',
        'detection/upload.html',
        'detection/result.html',
        'detection/history.html',
        'accounts/profile.html',
        'errors/404.html',
        'errors/500.html',
    ]
    
    for template in required_templates:
        template_path = os.path.join(templates_dir, template)
        if os.path.exists(template_path):
            print(f"✓ {template}")
        else:
            print(f"✗ {template} - Missing!")

def check_models():
    """Check if ML models exist"""
    print("\n" + "="*50)
    print("Checking ML Models")
    print("="*50)
    
    outputs_dir = os.path.join(project_root, 'outputs')
    
    models = [
        'vgg16_oral_cancer_model.pth',
        'regnet_oral_cancer_model.pth',
    ]
    
    for model in models:
        model_path = os.path.join(outputs_dir, model)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"✓ {model} ({size:.1f} MB)")
        else:
            print(f"✗ {model} - Missing! (Detection may not work)")

if __name__ == '__main__':
    print("OralScan AI - Web UI Testing")
    print("="*50)
    
    test_all_urls()
    check_templates()
    check_models()
    
    print("\n" + "="*50)
    print("Testing Complete!")
    print("="*50)
    print("\nTo fix any issues:")
    print("1. Missing templates: Check if files exist in templates/")
    print("2. 404 errors: Check URL patterns in urls.py files")
    print("3. 500 errors: Check views.py for import errors")
    print("4. Missing models: Train or download ML models")
    print("\nRun server with: python scripts/run_server.py")