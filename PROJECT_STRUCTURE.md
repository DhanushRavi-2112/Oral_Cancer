# Oral Cancer Detection Project Structure

## Directory Layout

```
Oral_cancer/
├── apps/                       # Django applications
│   ├── accounts/              # User authentication and profiles
│   ├── dashboard/             # Main dashboard and stats
│   ├── detection/             # Cancer detection logic and ML integration
│   └── pages/                 # Static pages (home, about, contact)
│
├── config/                    # Configuration files
│   ├── oralcancer_web/       # Django project settings
│   │   ├── __init__.py
│   │   ├── asgi.py           # ASGI configuration
│   │   ├── settings.py       # Django settings
│   │   ├── urls.py           # Root URL configuration
│   │   └── wsgi.py           # WSGI configuration
│   ├── requirements.txt       # Python dependencies
│   ├── requirements_django.txt
│   └── django_requirements.txt
│
├── data/                      # Dataset directory (gitignored)
│   ├── train/
│   ├── val/
│   └── test/
│
├── dataset/                   # Raw dataset files (gitignored)
│
├── docs/                      # Documentation
│   ├── README.md
│   ├── DJANGO_SETUP_GUIDE.md
│   ├── PRODUCTION_DEPLOYMENT_GUIDE.md
│   └── ...
│
├── media/                     # User uploaded files (gitignored)
│
├── ml_scripts/               # Machine learning scripts
│   ├── train_regnet.py
│   ├── train_optimized_regnet.py
│   ├── predict.py
│   ├── predict_dual_model.py
│   ├── convert_regnet_model.py
│   └── organize_dataset.py
│
├── outputs/                   # Model outputs and results (gitignored)
│   ├── models/
│   ├── logs/
│   └── figures/
│
├── scripts/                   # Utility scripts
│   ├── setup_django.py
│   ├── fix_admin_access.py
│   ├── *.bat                 # Windows batch scripts
│   └── *.sh                  # Unix shell scripts
│
├── src/                       # Core ML source code
│   ├── data/
│   │   └── preprocessing.py
│   ├── evaluation/
│   │   └── compare_models.py
│   ├── models/
│   │   ├── architectures.py
│   │   └── architectures_optimized.py
│   └── training/
│       └── train.py
│
├── static/                    # Static assets
│   ├── css/
│   └── js/
│
├── templates/                 # Django HTML templates
│   ├── base.html
│   ├── accounts/
│   ├── dashboard/
│   ├── detection/
│   ├── pages/
│   └── registration/
│
├── tests/                     # Test files
│   ├── unit/
│   ├── integration/
│   └── *.py
│
├── venv/                      # Virtual environment (gitignored)
│
├── .gitignore                 # Git ignore file
├── db.sqlite3                 # SQLite database (gitignored)
├── manage.py                  # Django management script
├── pyproject.toml            # Python project configuration
└── requirements.txt          # Main requirements file
```

## Key Components

### 1. Django Applications (`apps/`)
- **accounts**: User management, authentication, profiles
- **dashboard**: Main interface, statistics, user dashboard
- **detection**: Core cancer detection functionality, ML model integration
- **pages**: Static pages like home, about, contact

### 2. Configuration (`config/`)
- Django project settings and URL routing
- Requirements files for different environments
- WSGI/ASGI configurations for deployment

### 3. Machine Learning (`src/` and `ml_scripts/`)
- **src/**: Core ML library code (models, training, evaluation)
- **ml_scripts/**: Executable scripts for training and inference

### 4. Data Directories
- **data/**: Organized dataset for training/validation/testing
- **dataset/**: Raw data files
- **media/**: User uploaded images
- **outputs/**: Model checkpoints, logs, and results

### 5. Web Assets
- **static/**: CSS, JavaScript, and other static files
- **templates/**: Django HTML templates

### 6. Development Tools
- **scripts/**: Setup and utility scripts
- **tests/**: Unit and integration tests
- **docs/**: Project documentation

## Import Structure

```python
# Django apps
from apps.detection.models import Detection
from apps.accounts.views import profile_view

# ML components
from src.models.architectures import create_regnet_model
from src.data.preprocessing import preprocess_image

# Configuration
from config.oralcancer_web.settings import BASE_DIR
```

## Environment Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run migrations:
   ```bash
   python manage.py migrate
   ```

4. Create superuser:
   ```bash
   python manage.py createsuperuser
   ```

5. Run development server:
   ```bash
   python manage.py runserver
   ```

## Best Practices

1. **Code Organization**:
   - Keep Django apps focused on single responsibilities
   - Separate ML logic from web logic
   - Use appropriate directory for each file type

2. **Configuration**:
   - Use environment variables for sensitive data
   - Keep settings modular for different environments
   - Document all configuration options

3. **Data Management**:
   - Never commit large datasets or model files
   - Use Git LFS for necessary large files
   - Keep data paths configurable

4. **Testing**:
   - Write tests for both Django views and ML functions
   - Separate unit tests from integration tests
   - Use pytest for testing framework

5. **Documentation**:
   - Keep README files updated
   - Document API endpoints
   - Include setup instructions