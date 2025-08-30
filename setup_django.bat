@echo off
echo Setting up Django environment...

REM Install Django dependencies (core only)
pip install -r requirements_clean.txt

REM Create Django migrations
python manage.py makemigrations
python manage.py makemigrations accounts
python manage.py makemigrations detection
python manage.py makemigrations dashboard
python manage.py makemigrations pages

REM Apply migrations
python manage.py migrate

REM Create superuser (optional)
echo.
echo To create a superuser, run: python manage.py createsuperuser

REM Collect static files
python manage.py collectstatic --noinput

echo.
echo Django setup complete! Run the server with:
echo python manage.py runserver
pause