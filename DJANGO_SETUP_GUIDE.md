# Django Web Application Setup Guide

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install Django and dependencies
pip install -r requirements_django.txt
```

### 2. Create Django Project Structure
```bash
# Run the setup script
python setup_django.py

# This will create the project structure and install all required packages
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
```

### 4. Initialize Database
```bash
# Create database migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser account
python manage.py createsuperuser

# Load initial data (optional)
python manage.py loaddata initial_data.json
```

### 5. Collect Static Files
```bash
python manage.py collectstatic --noinput
```

### 6. Run Development Server
```bash
python manage.py runserver
```

Visit `http://localhost:8000` to see your application!

## 📁 Project Structure

```
oralcancer_web/
├── manage.py              # Django management script
├── oralcancer_web/        # Main project directory
│   ├── settings.py        # Project settings
│   ├── urls.py           # Main URL configuration
│   └── wsgi.py           # WSGI configuration
├── accounts/             # User authentication app
├── detection/            # Core detection functionality
├── dashboard/            # User dashboard
├── pages/               # Static pages (home, about, etc.)
├── templates/           # HTML templates
├── static/              # CSS, JS, images
├── media/               # User uploaded files
└── requirements_django.txt
```

## 🎨 Key Features Implemented

### 1. **Authentication System**
- User registration with email verification
- Login/Logout functionality
- Password reset via email
- Social authentication ready (Google, GitHub)

### 2. **Detection System**
- Image upload with drag-and-drop
- Multiple model selection (VGG16, RegNet, Ensemble)
- Real-time analysis progress
- Detailed results with confidence scores
- Detection history tracking

### 3. **Dashboard**
- User statistics and analytics
- Detection history with filters
- Profile management
- Settings customization

### 4. **Professional UI/UX**
- Responsive Bootstrap 5 design
- Modern gradient aesthetics
- Smooth animations
- Mobile-friendly interface
- Dark mode support (can be added)

### 5. **Clinical Features**
- Clinical notes for each detection
- Report generation
- Risk factor assessment
- Follow-up scheduling
- Multi-user collaboration

## 🔧 Development Commands

### Database Management
```bash
# Make migrations after model changes
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create admin user
python manage.py createsuperuser

# Reset database
python manage.py flush
```

### Static Files
```bash
# Collect static files
python manage.py collectstatic

# Compress static files (requires django-compressor)
python manage.py compress
```

### Testing
```bash
# Run all tests
python manage.py test

# Run specific app tests
python manage.py test detection

# Run with coverage
coverage run --source='.' manage.py test
coverage report
```

## 🚀 Production Deployment

### 1. Update Settings
```python
# In settings.py for production
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com']

# Use PostgreSQL
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'oralcancer_db',
        'USER': 'your_db_user',
        'PASSWORD': 'your_db_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# Security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

### 2. Deploy with Gunicorn
```bash
# Install Gunicorn
pip install gunicorn

# Run Gunicorn
gunicorn oralcancer_web.wsgi:application --bind 0.0.0.0:8000
```

### 3. Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location = /favicon.ico { access_log off; log_not_found off; }
    location /static/ {
        root /path/to/your/project;
    }
    location /media/ {
        root /path/to/your/project;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/gunicorn.sock;
    }
}
```

### 4. Systemd Service
Create `/etc/systemd/system/gunicorn.service`:
```ini
[Unit]
Description=gunicorn daemon
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/project
ExecStart=/path/to/venv/bin/gunicorn \
          --access-logfile - \
          --workers 3 \
          --bind unix:/run/gunicorn.sock \
          oralcancer_web.wsgi:application

[Install]
WantedBy=multi-user.target
```

## 🔒 Security Checklist

- [ ] Change SECRET_KEY for production
- [ ] Set DEBUG=False
- [ ] Configure ALLOWED_HOSTS
- [ ] Use HTTPS (SSL certificate)
- [ ] Set secure cookie flags
- [ ] Enable CSRF protection
- [ ] Use environment variables for secrets
- [ ] Regular security updates
- [ ] Implement rate limiting
- [ ] Add Content Security Policy headers

## 📱 API Endpoints

### Detection API
```
POST /detection/api/predict/
Content-Type: multipart/form-data
Authorization: Token your-auth-token

Parameters:
- image: Image file
- model: 'vgg16', 'regnet', or 'ensemble'
```

### User API
```
GET /api/user/profile/
GET /api/user/detections/
POST /api/user/update-profile/
```

## 🛠️ Troubleshooting

### Common Issues

1. **Static files not loading**
   ```bash
   python manage.py collectstatic --noinput
   # Check STATIC_ROOT and STATIC_URL settings
   ```

2. **Database errors**
   ```bash
   python manage.py migrate --run-syncdb
   ```

3. **Import errors**
   ```bash
   # Ensure all apps are in INSTALLED_APPS
   # Check for circular imports
   ```

4. **Model not found errors**
   - Ensure ML models are in `outputs/models/` directory
   - Check MODEL_PATHS in settings.py

## 📞 Support

For issues or questions:
1. Check Django documentation: https://docs.djangoproject.com/
2. Review error logs in `logs/` directory
3. Enable DEBUG=True for detailed error pages

## 🎉 Next Steps

1. **Customize the UI** - Modify templates and CSS to match your brand
2. **Add more features** - Implement additional analysis tools
3. **Integrate payment** - Add Stripe for premium features
4. **Mobile app** - Create React Native app using the API
5. **Analytics** - Add Google Analytics or Plausible

Your professional medical web application is ready! 🚀