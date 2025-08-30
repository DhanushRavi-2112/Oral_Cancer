@echo off
echo 🔧 Quick Fix for Database Error

echo Creating database tables...
python manage.py makemigrations
python manage.py migrate

echo Creating media directory...
mkdir media 2>nul

echo ✅ Fixed! Now try accessing the upload page again.
pause