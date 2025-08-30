#!/usr/bin/env python3
"""
Comprehensive test of Django web application
Tests all functionality and provides detailed feedback
"""
import requests
import sys
from urllib.parse import urljoin

# Base URL for testing
BASE_URL = 'http://localhost:8000'

def test_public_page(url, description):
    """Test a public page that should return 200"""
    try:
        response = requests.get(urljoin(BASE_URL, url), timeout=10)
        if response.status_code == 200:
            print(f"✅ {description}: {url}")
            return True
        else:
            print(f"❌ {description}: {url} (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {description}: {url} (Error: {str(e)})")
        return False

def test_protected_page(url, description):
    """Test a protected page that should redirect to login (302)"""
    try:
        response = requests.get(urljoin(BASE_URL, url), allow_redirects=False, timeout=10)
        if response.status_code == 302:
            print(f"✅ {description}: {url} (Protected - redirects to login)")
            return True
        else:
            print(f"❌ {description}: {url} (Expected 302 redirect, got {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {description}: {url} (Error: {str(e)})")
        return False

def test_admin_page():
    """Test Django admin - should redirect to admin login"""
    try:
        response = requests.get(urljoin(BASE_URL, '/admin/'), allow_redirects=True, timeout=10)
        if response.status_code == 200 and 'admin' in response.url:
            print("✅ Django Admin: /admin/ (Shows admin login form)")
            return True
        else:
            print(f"❌ Django Admin: /admin/ (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Django Admin: /admin/ (Error: {str(e)})")
        return False

def main():
    print("🧪 Final Django Web Application Test")
    print("=" * 60)
    
    # Check server availability
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"🌐 Django server is running at {BASE_URL}\n")
    except requests.exceptions.RequestException:
        print(f"❌ Django server is not running at {BASE_URL}")
        print("Please start the server with: python manage.py runserver")
        sys.exit(1)
    
    passed = 0
    failed = 0
    
    print("📄 PUBLIC PAGES (Should return 200)")
    print("-" * 40)
    public_pages = [
        ('/', 'Home Page'),
        ('/about/', 'About Page'), 
        ('/contact/', 'Contact Page'),
        ('/accounts/login/', 'Login Page'),
        ('/accounts/signup/', 'Signup Page'),
        ('/detection/compare/', 'Model Comparison'),
    ]
    
    for url, desc in public_pages:
        if test_public_page(url, desc):
            passed += 1
        else:
            failed += 1
    
    print(f"\n🔒 PROTECTED PAGES (Should redirect to login)")
    print("-" * 40)
    protected_pages = [
        ('/dashboard/', 'Dashboard Home'),
        ('/dashboard/stats/', 'Dashboard Statistics'),
        ('/detection/upload/', 'Image Upload'),
        ('/detection/history/', 'Detection History'),
        ('/accounts/profile/', 'User Profile'),
    ]
    
    for url, desc in protected_pages:
        if test_protected_page(url, desc):
            passed += 1
        else:
            failed += 1
    
    print(f"\n🔧 ADMIN PANEL")
    print("-" * 40)
    if test_admin_page():
        passed += 1
    else:
        failed += 1
    
    print(f"\n📁 STATIC FILES")
    print("-" * 40)
    static_files = [
        ('/static/css/style.css', 'CSS Stylesheet'),
        ('/static/js/main.js', 'JavaScript File'),
    ]
    
    for url, desc in static_files:
        if test_public_page(url, desc):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Django web application is working perfectly!")
        print("\n🚀 Ready to use:")
        print("1. Visit http://localhost:8000 to see the home page")
        print("2. Sign up at http://localhost:8000/accounts/signup/")
        print("3. Upload images at http://localhost:8000/detection/upload/")
        print("4. View model comparison at http://localhost:8000/detection/compare/")
        print("5. Access admin at http://localhost:8000/admin/")
    else:
        print(f"\n⚠️  {failed} issues found. Please check the output above.")
        print("Most common fixes:")
        print("- Run: python manage.py migrate")
        print("- Run: python manage.py collectstatic")
        print("- Check file permissions on static/media directories")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())