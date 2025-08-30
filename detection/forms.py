from django import forms
from django.core.exceptions import ValidationError
from .models import Detection, Report
from PIL import Image
import io

class DetectionForm(forms.Form):
    """Form for uploading images for detection"""
    
    MODEL_CHOICES = [
        ('ensemble', 'Ensemble AI (Recommended - Highest Accuracy)'),
        ('vgg16', 'VGG16 (Fast & Reliable)'),
        ('regnet', 'RegNetY-320 (Efficient)'),
    ]
    
    image = forms.ImageField(
        label='Upload Image',
        help_text='Supported formats: JPG, PNG, BMP (Max 10MB)',
        widget=forms.FileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control'
        })
    )
    
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        initial='ensemble',
        widget=forms.RadioSelect(attrs={
            'class': 'form-check-input'
        })
    )
    
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Add any relevant clinical notes or observations...'
        })
    )
    
    def clean_image(self):
        image = self.cleaned_data['image']
        print(f"DEBUG: Validating image - Name: {image.name}, Size: {image.size} bytes")
        
        # Check file size (10MB limit)
        if image.size > 10 * 1024 * 1024:
            print(f"DEBUG: File too large: {image.size} bytes")
            raise ValidationError('Image file size cannot exceed 10MB.')
        
        # Validate image format only
        try:
            # Reset file position to beginning
            image.seek(0)
            img = Image.open(image)
            print(f"DEBUG: Image opened successfully - Format: {img.format}, Size: {img.size}")
            
            # Only validate that it's a proper image file - no dimension restrictions
            # The ML models will handle resizing automatically
            
            # Reset file position again for later use
            image.seek(0)
            print("DEBUG: Image validation passed successfully")
                
        except Exception as e:
            print(f"DEBUG: Image validation failed with error: {str(e)}")
            raise ValidationError('Invalid image file. Please upload a valid image.')
        
        return image


class ReportForm(forms.ModelForm):
    """Form for creating clinical reports"""
    
    class Meta:
        model = Report
        fields = [
            'clinical_findings', 
            'recommendations',
            'follow_up_required',
            'follow_up_date',
            'smoking_history',
            'alcohol_use',
            'family_history',
            'previous_lesions'
        ]
        
        widgets = {
            'clinical_findings': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe clinical findings and observations...'
            }),
            'recommendations': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Provide treatment recommendations...'
            }),
            'follow_up_date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date'
            }),
            'follow_up_required': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
            'smoking_history': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
            'alcohol_use': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
            'family_history': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
            'previous_lesions': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }
        
        labels = {
            'follow_up_required': 'Follow-up Required?',
            'follow_up_date': 'Follow-up Date',
            'smoking_history': 'History of Smoking',
            'alcohol_use': 'Regular Alcohol Use',
            'family_history': 'Family History of Oral Cancer',
            'previous_lesions': 'Previous Oral Lesions'
        }
    
    def clean(self):
        cleaned_data = super().clean()
        follow_up_required = cleaned_data.get('follow_up_required')
        follow_up_date = cleaned_data.get('follow_up_date')
        
        if follow_up_required and not follow_up_date:
            raise ValidationError('Please specify a follow-up date.')
        
        return cleaned_data


class FilterForm(forms.Form):
    """Form for filtering detection history"""
    
    FILTER_CHOICES = [
        ('all', 'All Results'),
        ('cancer', 'Cancer Detected'),
        ('healthy', 'Healthy'),
        ('reviewed', 'Reviewed'),
        ('unreviewed', 'Not Reviewed'),
    ]
    
    filter_type = forms.ChoiceField(
        choices=FILTER_CHOICES,
        required=False,
        initial='all',
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )
    
    date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date',
            'placeholder': 'From Date'
        })
    )
    
    date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date',
            'placeholder': 'To Date'
        })
    )
    
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search by patient name or notes...'
        })
    )