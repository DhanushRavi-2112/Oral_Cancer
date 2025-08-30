@echo off
echo Setting up Django database...

echo Step 1: Creating migrations for all apps...
python manage.py makemigrations accounts
python manage.py makemigrations detection
python manage.py makemigrations dashboard
python manage.py makemigrations pages

echo Step 2: Creating general migrations...
python manage.py makemigrations

echo Step 3: Applying all migrations...
python manage.py migrate

echo Step 4: Creating media directories...
mkdir media 2>nul
mkdir media\detections 2>nul
mkdir media\detections\2024 2>nul
mkdir media\detections\2025 2>nul

echo.
echo ✅ Database setup complete!
echo.
echo Next steps:
echo 1. Create superuser: python manage.py createsuperuser
echo 2. Start server: python manage.py runserver
echo.
pause