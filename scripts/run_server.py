#!/usr/bin/env python
"""
Run Django development server with proper configuration
"""
import os
import sys
import subprocess

def main():
    """Run the development server"""
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # Set Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.oralcancer_web.settings')
    
    # Check if migrations are needed
    print("Checking database migrations...")
    try:
        subprocess.run([sys.executable, 'manage.py', 'makemigrations', '--check'], 
                      cwd=project_root, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Creating migrations...")
        subprocess.run([sys.executable, 'manage.py', 'makemigrations'], cwd=project_root)
        print("Applying migrations...")
        subprocess.run([sys.executable, 'manage.py', 'migrate'], cwd=project_root)
    
    # Collect static files
    print("Collecting static files...")
    subprocess.run([sys.executable, 'manage.py', 'collectstatic', '--noinput'], 
                  cwd=project_root, capture_output=True)
    
    # Run the server
    print("\n" + "="*50)
    print("Starting OralScan AI Development Server")
    print("="*50)
    print("Access the application at: http://127.0.0.1:8000")
    print("Admin panel at: http://127.0.0.1:8000/admin")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    subprocess.run([sys.executable, 'manage.py', 'runserver'], cwd=project_root)

if __name__ == '__main__':
    main()