from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid

class Detection(models.Model):
    """Model to store oral cancer detection results"""
    
    RESULT_CHOICES = [
        ('cancer', 'Cancer Detected'),
        ('healthy', 'Healthy'),
    ]
    
    MODEL_CHOICES = [
        ('vgg16', 'VGG16'),
        ('regnet', 'RegNetY-320'),
        ('ensemble', 'Ensemble (Both Models)'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='detections')
    
    # Image data
    image = models.ImageField(upload_to='detections/%Y/%m/%d/')
    image_name = models.CharField(max_length=255)
    
    # Detection results
    result = models.CharField(max_length=10, choices=RESULT_CHOICES, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    cancer_probability = models.FloatField(null=True, blank=True)
    healthy_probability = models.FloatField(null=True, blank=True)
    
    # Model information
    model_used = models.CharField(max_length=20, choices=MODEL_CHOICES, default='ensemble')
    processing_time = models.FloatField(help_text="Processing time in seconds", null=True, blank=True)
    
    # Individual model results (for ensemble)
    vgg16_result = models.CharField(max_length=10, choices=RESULT_CHOICES, null=True, blank=True)
    vgg16_confidence = models.FloatField(null=True, blank=True)
    regnet_result = models.CharField(max_length=10, choices=RESULT_CHOICES, null=True, blank=True)
    regnet_confidence = models.FloatField(null=True, blank=True)
    
    # Uncertainty (for ensemble)
    uncertainty_level = models.CharField(max_length=10, null=True, blank=True)
    disagreement_score = models.FloatField(null=True, blank=True)
    
    # Clinical notes
    notes = models.TextField(blank=True)
    is_reviewed = models.BooleanField(default=False)
    reviewed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, 
                                   related_name='reviewed_detections')
    reviewed_at = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Detection'
        verbose_name_plural = 'Detections'
    
    def __str__(self):
        return f"{self.user.username} - {self.result} ({self.confidence:.1%})"
    
    def get_confidence_percentage(self):
        return f"{self.confidence * 100:.1f}%"
    
    def get_result_color(self):
        return 'danger' if self.result == 'cancer' else 'success'
    
    def get_uncertainty_badge(self):
        if not self.uncertainty_level:
            return ''
        
        badges = {
            'low': 'success',
            'medium': 'warning',
            'high': 'danger'
        }
        return badges.get(self.uncertainty_level, 'secondary')


class UserProfile(models.Model):
    """Extended user profile for medical professionals"""
    
    ROLE_CHOICES = [
        ('doctor', 'Doctor'),
        ('dentist', 'Dentist'),
        ('radiologist', 'Radiologist'),
        ('researcher', 'Researcher'),
        ('student', 'Student'),
        ('other', 'Other'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    # Professional information
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='doctor')
    institution = models.CharField(max_length=255, blank=True)
    license_number = models.CharField(max_length=100, blank=True)
    specialization = models.CharField(max_length=200, blank=True)
    
    # Contact information
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    
    # Settings
    notifications_enabled = models.BooleanField(default=True)
    email_reports = models.BooleanField(default=False)
    preferred_model = models.CharField(max_length=20, choices=Detection.MODEL_CHOICES, default='ensemble')
    
    # Statistics
    total_detections = models.IntegerField(default=0)
    cancer_detections = models.IntegerField(default=0)
    healthy_detections = models.IntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.role}"
    
    def update_statistics(self):
        """Update user statistics based on detections"""
        detections = self.user.detections.all()
        self.total_detections = detections.count()
        self.cancer_detections = detections.filter(result='cancer').count()
        self.healthy_detections = detections.filter(result='healthy').count()
        self.save()


class Report(models.Model):
    """Detailed reports for detections"""
    
    detection = models.OneToOneField(Detection, on_delete=models.CASCADE, related_name='report')
    
    # Clinical findings
    clinical_findings = models.TextField()
    recommendations = models.TextField()
    follow_up_required = models.BooleanField(default=False)
    follow_up_date = models.DateField(null=True, blank=True)
    
    # Risk factors
    smoking_history = models.BooleanField(default=False)
    alcohol_use = models.BooleanField(default=False)
    family_history = models.BooleanField(default=False)
    previous_lesions = models.BooleanField(default=False)
    
    # Report metadata
    generated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    generated_at = models.DateTimeField(default=timezone.now)
    pdf_report = models.FileField(upload_to='reports/%Y/%m/', null=True, blank=True)
    
    def __str__(self):
        return f"Report for Detection {self.detection.id}"