@echo off
echo Testing Django setup...
echo.

REM Test Django installation
python -c "import django; print('Django version:', django.get_version())"
if errorlevel 1 (
    echo ERROR: Django is not installed!
    echo Run: pip install Django==4.2.7 Pillow==9.5.0 whitenoise==6.6.0
    pause
    exit /b 1
)

echo ✓ Django is installed

REM Test settings import
python -c "import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'oralcancer_web.settings'); import django; django.setup(); print('✓ Settings loaded successfully')"
if errorlevel 1 (
    echo ERROR: Settings import failed!
    pause
    exit /b 1
)

REM Test makemigrations (dry run)
echo.
echo Testing makemigrations...
python manage.py makemigrations --dry-run
if errorlevel 1 (
    echo ERROR: Makemigrations failed!
    pause
    exit /b 1
)

echo.
echo ✓ Django setup test passed!
echo You can now run:
echo   python manage.py makemigrations
echo   python manage.py migrate
echo   python manage.py createsuperuser
echo   python manage.py runserver
echo.
pause