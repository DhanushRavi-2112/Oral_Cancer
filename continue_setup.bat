@echo off
echo Continuing Django setup...

echo Step 1: Creating migrations...
python manage.py makemigrations

echo.
echo Step 2: Applying migrations...
python manage.py migrate

echo.
echo Step 3: Creating superuser (optional)...
echo You can skip this step if you want
python manage.py createsuperuser

echo.
echo Step 4: Starting development server...
echo Visit: http://localhost:8000
python manage.py runserver

pause