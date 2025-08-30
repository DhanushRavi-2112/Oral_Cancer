@echo off
echo ========================================
echo OralScan AI - Project Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist "media" mkdir media
if not exist "static_root" mkdir static_root
if not exist "outputs" mkdir outputs
if not exist "logs" mkdir logs

REM Add project to PYTHONPATH
set PYTHONPATH=%cd%;%PYTHONPATH%

REM Run migrations
echo Running database migrations...
python manage.py makemigrations
python manage.py migrate

REM Collect static files
echo Collecting static files...
python manage.py collectstatic --noinput

REM Create superuser
echo.
echo ========================================
echo Create a superuser account for admin access
echo ========================================
python manage.py createsuperuser

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To run the server:
echo   run_server.bat
echo.
echo Or manually:
echo   venv\Scripts\activate
echo   python manage.py runserver
echo.
pause