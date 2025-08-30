from django.shortcuts import render


def home_view(request):
    """Home page view"""
    return render(request, 'pages/home.html')


def about_view(request):
    """About page view"""
    return render(request, 'pages/about.html')


def contact_view(request):
    """Contact page view"""
    return render(request, 'pages/contact.html')


def error_404(request, exception):
    """Custom 404 error page"""
    return render(request, 'errors/404.html', status=404)


def error_500(request):
    """Custom 500 error page"""
    return render(request, 'errors/500.html', status=500)