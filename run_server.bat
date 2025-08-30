@echo off
echo ========================================
echo Starting OralScan AI Development Server
echo ========================================
echo.

REM Add project root to PYTHONPATH
set PYTHONPATH=%cd%;%PYTHONPATH%

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Activating virtual environment...
    call venv\Scripts\activate
)

echo Checking database migrations...
python manage.py migrate

echo.
echo Collecting static files...
python manage.py collectstatic --noinput >nul 2>&1

echo.
echo ========================================
echo Server starting at http://127.0.0.1:8000
echo Admin panel at http://127.0.0.1:8000/admin
echo Press Ctrl+C to stop
echo ========================================
echo.

python manage.py runserver