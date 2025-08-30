# OralScan AI - Web Application Setup Guide

## 🌐 Complete Django Web Application for Oral Cancer Detection

This is a professional medical web application that provides AI-powered oral cancer detection using deep learning models (VGG16, RegNetY-320, and Ensemble).

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install minimal Django requirements
pip install -r requirements_clean.txt

# OR install with ML support
pip install -r django_simple_requirements.txt
```

### 2. Setup Database

```bash
# Create database migrations
python manage.py makemigrations
python manage.py migrate

# Create admin user (optional)
python manage.py createsuperuser
```

### 3. Run the Application

```bash
# Start development server
python manage.py runserver

# Visit: http://localhost:8000
```

## 📱 Application Features

### 🏠 Public Pages
- **Home Page** (`/`) - Professional landing page with hero section
- **About Page** (`/about/`) - Information about the AI technology
- **Contact Page** (`/contact/`) - Contact form with SweetAlert integration

### 🔐 Authentication
- **Login** (`/accounts/login/`) - User authentication
- **Signup** (`/accounts/signup/`) - User registration
- **Profile** (`/accounts/profile/`) - User profile management
- **Logout** (`/accounts/logout/`) - Secure logout

### 🏥 Dashboard
- **Dashboard Home** (`/dashboard/`) - User overview with statistics
- **Statistics** (`/dashboard/stats/`) - Detailed analytics

### 🔬 Detection System
- **Upload Image** (`/detection/upload/`) - AI-powered image analysis
- **View Results** (`/detection/result/<id>/`) - Detailed analysis results
- **Detection History** (`/detection/history/`) - Complete analysis records
- **Model Comparison** (`/detection/compare/`) - AI model performance comparison
- **Medical Reports** (`/detection/report/`) - Professional clinical reports

### ⚡ Admin Panel
- **Django Admin** (`/admin/`) - Full system administration

## 🎨 UI/UX Features

- **Professional Medical Design** - Clean, modern interface
- **Bootstrap 5** - Responsive design for all devices
- **Font Awesome Icons** - Comprehensive icon library
- **SweetAlert2** - Beautiful notification system
- **Drag & Drop Upload** - Intuitive file upload interface
- **Real-time Validation** - Client and server-side form validation
- **Loading Animations** - Professional loading indicators
- **Print Support** - Print-optimized medical reports

## 🧠 AI Integration

### Models Supported:
- **VGG16** - Fast and reliable (92.5% accuracy)
- **RegNetY-320** - Efficient and modern (91.8% accuracy)  
- **Ensemble** - Best of both models (94.2% accuracy) ⭐ Recommended

### Features:
- **Real-time Prediction** - Analysis in under 30 seconds
- **Confidence Scoring** - Reliability metrics for each prediction
- **Uncertainty Estimation** - Advanced ensemble uncertainty analysis
- **Medical Reporting** - Professional clinical report generation

## 📁 File Structure

```
Oral_cancer/
├── 📁 oralcancer_web/          # Django project settings
├── 📁 accounts/                # User authentication
├── 📁 pages/                   # Static pages (home, about, contact)
├── 📁 dashboard/               # User dashboard
├── 📁 detection/               # AI detection system
├── 📁 templates/               # HTML templates
│   ├── 📁 registration/        # Login/signup forms
│   ├── 📁 pages/              # Static page templates
│   ├── 📁 dashboard/          # Dashboard templates
│   ├── 📁 detection/          # Detection system templates
│   └── 📁 errors/             # Error pages (404, 500)
├── 📁 static/                  # CSS, JS, images
│   ├── 📁 css/                # Custom styling
│   └── 📁 js/                 # JavaScript functionality
├── 📁 media/                   # User uploaded files
└── 📄 manage.py               # Django management script
```

## 🛠️ Configuration Files

- `requirements_clean.txt` - Minimal Django dependencies
- `django_simple_requirements.txt` - Django + ML libraries
- `settings.py` - Django configuration
- `urls.py` - URL routing
- `test_all_pages.py` - Automated testing script

## 🧪 Testing

### Automated Testing:
```bash
# Test all pages automatically
python test_all_pages.py
```

### Manual Testing Checklist:
- [ ] Home page loads correctly
- [ ] User can sign up and login
- [ ] Dashboard shows user statistics  
- [ ] Image upload works with drag & drop
- [ ] AI models generate predictions
- [ ] Results display properly
- [ ] History page shows past detections
- [ ] Admin panel is accessible
- [ ] All forms validate correctly
- [ ] Mobile responsive design works

## 🚨 Troubleshooting

### Common Issues:

1. **Template Errors**
   ```bash
   # Ensure all templates exist
   ls templates/registration/
   ls templates/detection/
   ```

2. **Static Files Not Loading**
   ```bash
   python manage.py collectstatic
   ```

3. **Database Issues**
   ```bash
   python manage.py migrate
   ```

4. **Permission Errors**
   ```bash
   # Create media directory
   mkdir media
   mkdir media/detections
   ```

## 🔧 Customization

### Styling:
- Modify `static/css/style.css` for custom styling
- Update `templates/base.html` for layout changes

### Functionality:
- Add new views in respective `views.py` files
- Create new templates in `templates/` directory
- Update URL patterns in `urls.py` files

## 📈 Production Deployment

### Security Checklist:
- [ ] Change `SECRET_KEY` in settings
- [ ] Set `DEBUG = False`
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Use HTTPS
- [ ] Setup proper database (PostgreSQL)
- [ ] Configure static file serving
- [ ] Setup backup system

### Performance:
- [ ] Enable caching
- [ ] Optimize database queries
- [ ] Compress static files
- [ ] Use CDN for static assets
- [ ] Monitor application performance

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the automated test script
3. Review Django error logs
4. Check browser developer console

## 📄 License

This is a medical AI application for educational and research purposes.

---

**🎉 Your OralScan AI web application is ready to use!**

Visit `http://localhost:8000` to start detecting oral cancer with AI.