# OralScan AI - Quick Start Guide

## 🚀 Running the Application

### Prerequisites
- Python 3.8+ installed
- Virtual environment activated
- All dependencies installed

### Quick Start Commands

```bash
# 1. Activate virtual environment (if not already active)
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate

# 2. Run database migrations
python manage.py migrate

# 3. Create superuser (admin access)
python manage.py createsuperuser

# 4. Collect static files
python manage.py collectstatic --noinput

# 5. Run the development server
python manage.py runserver
```

### Access the Application

- **Main Application**: http://127.0.0.1:8000
- **Admin Panel**: http://127.0.0.1:8000/admin

### Test Credentials (if using demo data)
- Username: `testuser`
- Password: `testpass123`

## 🛠️ Troubleshooting

### Common Issues and Solutions

1. **Import Errors**
   ```bash
   # Fix Python path issues
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Static Files Not Loading**
   ```bash
   python manage.py collectstatic --noinput
   ```

3. **Database Errors**
   ```bash
   # Reset database
   rm db.sqlite3
   python manage.py migrate
   python manage.py createsuperuser
   ```

4. **ML Models Not Found**
   - Ensure model files exist in `outputs/` directory
   - Download pre-trained models or train new ones

### Run Complete Test
```bash
python scripts/test_web_ui.py
```

## 📱 Key Features

1. **Home Page** - Landing page with feature overview
2. **Sign Up/Login** - User authentication system
3. **Upload Image** - Drag & drop interface for image upload
4. **AI Analysis** - Real-time cancer detection
5. **Results View** - Detailed analysis with confidence scores
6. **History** - Track all previous detections
7. **Dashboard** - User statistics and insights

## 🔧 Development Tips

### Run with Debug Toolbar
```bash
pip install django-debug-toolbar
python manage.py runserver --settings=config.oralcancer_web.settings_dev
```

### Run Tests
```bash
python manage.py test
```

### Check Code Quality
```bash
flake8 apps/
black apps/ --check
```

## 📝 Notes

- The application uses SQLite for development (automatic setup)
- ML models need to be trained or downloaded separately
- Images are stored in `media/` directory
- Static files are served from `static/` directory

## 🆘 Need Help?

1. Check the logs in the terminal
2. Review `docs/` directory for detailed documentation
3. Run `python scripts/test_web_ui.py` to diagnose issues
4. Check if all files are in correct directories per `PROJECT_STRUCTURE.md`

---

**Quick Commands Summary:**
```bash
# Start server
python manage.py runserver

# Run tests
python scripts/test_web_ui.py

# Create admin user
python manage.py createsuperuser
```