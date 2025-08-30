@echo off
echo Installing Django (minimal setup)...

REM Install only essential packages one by one
pip install Django==4.2.7
pip install Pillow==9.5.0
pip install whitenoise==6.6.0

echo.
echo Django installation complete!
echo.
echo Now run the setup:
echo python manage.py makemigrations
echo python manage.py migrate
echo python manage.py createsuperuser
echo python manage.py runserver
echo.
pause