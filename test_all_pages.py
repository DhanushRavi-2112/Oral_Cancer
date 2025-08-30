#!/usr/bin/env python3
"""
Test script to verify all Django pages are working
Run this after starting the Django server
"""
import requests
import sys

# Base URL for testing
BASE_URL = 'http://localhost:8000'

# Pages to test (URL, Expected Status Code, Description)
TEST_PAGES = [
    # Public pages
    ('/', 200, 'Home page'),
    ('/about/', 200, 'About page'),
    ('/contact/', 200, 'Contact page'),
    
    # Authentication pages
    ('/accounts/login/', 200, 'Login page'),
    ('/accounts/signup/', 200, 'Signup page'),
    
    # Protected pages (should redirect to login)
    ('/dashboard/', 302, 'Dashboard (redirect to login)'),
    ('/detection/upload/', 302, 'Upload page (redirect to login)'),
    ('/detection/history/', 302, 'History page (redirect to login)'),
    ('/detection/compare/', 200, 'Model comparison page'),
    ('/accounts/profile/', 302, 'Profile page (redirect to login)'),
    
    # Admin page (redirects to login)
    ('/admin/', 302, 'Django admin page (redirect to login)'),
    
    # Static files
    ('/static/css/style.css', 200, 'CSS file'),
    ('/static/js/main.js', 200, 'JavaScript file'),
]

def test_page(url, expected_status, description):
    """Test a single page"""
    try:
        full_url = BASE_URL + url
        response = requests.get(full_url, allow_redirects=False, timeout=10)
        
        if response.status_code == expected_status:
            print(f"✅ {description}: {url} (Status: {response.status_code})")
            return True
        else:
            print(f"❌ {description}: {url} (Expected: {expected_status}, Got: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {description}: {url} (Error: {str(e)})")
        return False

def main():
    print("🧪 Testing Django Web Application Pages")
    print("=" * 50)
    
    # Test server availability
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"🌐 Django server is running at {BASE_URL}")
    except requests.exceptions.RequestException:
        print(f"❌ Django server is not running at {BASE_URL}")
        print("Please start the server with: python manage.py runserver")
        sys.exit(1)
    
    print()
    
    # Test all pages
    passed = 0
    failed = 0
    
    for url, expected_status, description in TEST_PAGES:
        if test_page(url, expected_status, description):
            passed += 1
        else:
            failed += 1
    
    print()
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All pages are working correctly!")
        return 0
    else:
        print(f"⚠️  {failed} pages have issues. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())