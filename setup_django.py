#!/usr/bin/env python3
"""
Setup Django project for Oral Cancer Detection Web Application
"""
import os
import sys
import subprocess

def create_django_structure():
    """Create Django project structure"""
    print("Setting up Django Web Application for Oral Cancer Detection")
    print("="*60)
    
    # Install Django and required packages
    packages = [
        'django>=4.2',
        'django-crispy-forms',
        'crispy-bootstrap5',
        'pillow',
        'django-allauth',  # For advanced authentication
        'django-widget-tweaks',
        'django-extensions',
        'python-decouple',  # For environment variables
        'whitenoise',  # For static files in production
        'gunicorn',  # Production server
        'psycopg2-binary',  # PostgreSQL support
    ]
    
    print("Installing required packages...")
    for package in packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', package])
    
    # Create Django project
    print("\nCreating Django project...")
    subprocess.run(['django-admin', 'startproject', 'oralcancer_web', '.'])
    
    # Create apps
    apps = ['detection', 'accounts', 'dashboard', 'pages']
    for app in apps:
        print(f"Creating app: {app}")
        subprocess.run([sys.executable, 'manage.py', 'startapp', app])
    
    print("\nDjango project structure created successfully!")
    print("\nNext steps:")
    print("1. Run: python manage.py migrate")
    print("2. Run: python manage.py createsuperuser")
    print("3. Run: python manage.py runserver")

if __name__ == "__main__":
    create_django_structure()