from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.conf import settings
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
import os
import json
import time
from PIL import Image
import io

from .models import Detection
from .forms import DetectionForm, ReportForm
from .ml_models import get_predictor


@login_required
def upload_image(request):
    """Upload and analyze image for oral cancer detection"""
    if request.method == 'POST':
        form = DetectionForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image temporarily
            image = form.cleaned_data['image']
            model_choice = form.cleaned_data['model_choice']
            
            # Create detection instance
            detection = Detection(
                user=request.user,
                image=image,
                image_name=image.name,
                model_used=model_choice
            )
            
            # Save image first to get the path
            detection.save()
            
            try:
                # Perform prediction
                image_path = detection.image.path
                start_time = time.time()
                
                # Use our integrated ML models
                predictions = get_predictor().predict_image(image_path)
                
                if 'error' in predictions:
                    raise Exception(predictions['error'])
                
                if model_choice == 'ensemble' and 'ensemble' in predictions:
                    result = predictions['ensemble']
                    # Store individual model results
                    if 'vgg16' in predictions:
                        detection.vgg16_result = predictions['vgg16']['prediction']
                        detection.vgg16_confidence = predictions['vgg16']['confidence']
                    if 'regnet' in predictions:
                        detection.regnet_result = predictions['regnet']['prediction']
                        detection.regnet_confidence = predictions['regnet']['confidence']
                elif model_choice == 'vgg16' and 'vgg16' in predictions:
                    result = predictions['vgg16']
                elif model_choice == 'regnet' and 'regnet' in predictions:
                    result = predictions['regnet']
                else:
                    # Fallback to any available model
                    result = next(iter(predictions.values()))
                
                # Update detection with results
                detection.result = result['prediction'].lower()
                detection.confidence = result['confidence']
                detection.cancer_probability = result['probability'] if result['prediction'] == 'Cancer' else 1 - result['probability']
                detection.healthy_probability = 1 - result['probability'] if result['prediction'] == 'Cancer' else result['probability']
                detection.processing_time = time.time() - start_time
                detection.save()
                
                messages.success(request, 'Image analyzed successfully!')
                return redirect('detection:result', detection_id=detection.id)
                
            except Exception as e:
                detection.delete()  # Clean up on error
                messages.error(request, f'Error during analysis: {str(e)}')
                return redirect('detection:upload')
    else:
        form = DetectionForm()
    
    context = {
        'form': form,
        'recent_detections': request.user.detections.all()[:5]
    }
    return render(request, 'detection/upload.html', context)


@login_required
def view_result(request, detection_id):
    """View detection result"""
    detection = get_object_or_404(Detection, id=detection_id, user=request.user)
    
    # Generate recommendations based on result
    if detection.result == 'cancer':
        recommendations = [
            "Immediate consultation with an oral surgeon recommended",
            "Biopsy required for definitive diagnosis",
            "Avoid tobacco and alcohol",
            "Schedule follow-up within 1 week"
        ]
    else:
        recommendations = [
            "Continue regular oral health checkups",
            "Maintain good oral hygiene",
            "Annual screening recommended",
            "Monitor for any changes"
        ]
    
    context = {
        'detection': detection,
        'recommendations': recommendations
    }
    return render(request, 'detection/result.html', context)


@login_required
def detection_history(request):
    """View all detection history"""
    detections = request.user.detections.all()
    
    # Filter options
    filter_type = request.GET.get('filter', 'all')
    if filter_type == 'cancer':
        detections = detections.filter(result='cancer')
    elif filter_type == 'healthy':
        detections = detections.filter(result='healthy')
    elif filter_type == 'reviewed':
        detections = detections.filter(is_reviewed=True)
    
    # Date range filter
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    if date_from:
        detections = detections.filter(created_at__gte=date_from)
    if date_to:
        detections = detections.filter(created_at__lte=date_to)
    
    context = {
        'detections': detections,
        'filter_type': filter_type,
        'total_count': request.user.detections.count(),
        'cancer_count': request.user.detections.filter(result='cancer').count(),
        'healthy_count': request.user.detections.filter(result='healthy').count(),
    }
    return render(request, 'detection/history.html', context)


@login_required
def create_report(request, detection_id):
    """Create a detailed report for a detection"""
    detection = get_object_or_404(Detection, id=detection_id)
    
    # Check if user is authorized (owner or medical professional)
    if detection.user != request.user and not request.user.is_staff:
        messages.error(request, 'You are not authorized to create a report for this detection.')
        return redirect('detection:result', detection_id=detection_id)
    
    if request.method == 'POST':
        form = ReportForm(request.POST)
        if form.is_valid():
            report = form.save(commit=False)
            report.detection = detection
            report.generated_by = request.user
            report.save()
            
            # Mark detection as reviewed
            detection.is_reviewed = True
            detection.reviewed_by = request.user
            detection.reviewed_at = timezone.now()
            detection.save()
            
            messages.success(request, 'Report created successfully!')
            return redirect('detection:view_report', detection_id=detection_id)
    else:
        form = ReportForm()
    
    context = {
        'form': form,
        'detection': detection
    }
    return render(request, 'detection/create_report.html', context)


@login_required
def view_report(request, detection_id):
    """View report for a detection"""
    detection = get_object_or_404(Detection, id=detection_id)
    
    # Check authorization
    if detection.user != request.user and not request.user.is_staff:
        messages.error(request, 'You are not authorized to view this report.')
        return redirect('dashboard:home')
    
    if not hasattr(detection, 'report'):
        messages.info(request, 'No report available for this detection.')
        return redirect('detection:result', detection_id=detection_id)
    
    context = {
        'detection': detection,
        'report': detection.report
    }
    return render(request, 'detection/view_report.html', context)


@login_required
def download_report(request, detection_id):
    """Download PDF report for a detection"""
    detection = get_object_or_404(Detection, id=detection_id, user=request.user)
    
    if not hasattr(detection, 'report'):
        messages.error(request, 'No report available for download.')
        return redirect('detection:result', detection_id=detection_id)
    
    # Generate PDF report (implement PDF generation logic)
    # For now, redirect to view report
    return redirect('detection:view_report', detection_id=detection_id)


@csrf_exempt
@login_required
def api_predict(request):
    """API endpoint for predictions (for AJAX requests)"""
    if request.method == 'POST':
        try:
            image_file = request.FILES.get('image')
            model_choice = request.POST.get('model', 'ensemble')
            
            if not image_file:
                return JsonResponse({'error': 'No image provided'}, status=400)
            
            # Validate image
            try:
                image_file.seek(0)
                img = Image.open(image_file)
                # Don't verify as it corrupts the file - just check if it opens
                img.load()  # This will raise an exception for invalid images
                image_file.seek(0)  # Reset for later use
            except:
                return JsonResponse({'error': 'Invalid image file'}, status=400)
            
            # Create temporary detection
            detection = Detection.objects.create(
                user=request.user,
                image=image_file,
                image_name=image_file.name,
                model_used=model_choice
            )
            
            # Perform prediction (similar to upload_image view)
            # ... prediction logic ...
            
            return JsonResponse({
                'success': True,
                'detection_id': str(detection.id),
                'result': detection.result,
                'confidence': detection.confidence,
                'redirect_url': f'/detection/result/{detection.id}/'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def model_comparison(request):
    """Compare different model performances - public page"""
    # This is a public information page showing model performance
    context = {
        'comparison_data': {
            'vgg16_accuracy': 0.925,  # From your evaluation
            'regnet_accuracy': 0.918,
            'ensemble_accuracy': 0.942
        }
    }
    return render(request, 'detection/model_comparison.html', context)