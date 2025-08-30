from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required
def dashboard_home(request):
    """Dashboard home view"""
    context = {
        'user': request.user
    }
    return render(request, 'dashboard/home.html', context)


@login_required  
def dashboard_stats(request):
    """Dashboard statistics view"""
    context = {
        'user': request.user
    }
    return render(request, 'dashboard/stats.html', context)