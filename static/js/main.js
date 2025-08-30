// OralScan AI - Main JavaScript Functions

document.addEventListener('DOMContentLoaded', function() {
    // Initialize navbar scroll effect
    initNavbarScroll();
    
    // Initialize back to top button
    initBackToTop();
    
    // Initialize smooth scrolling
    initSmoothScrolling();
    
    // Initialize form enhancements
    initFormEnhancements();
});

// Navbar scroll effect
function initNavbarScroll() {
    const navbar = document.getElementById('mainNav');
    if (!navbar) return;
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('navbar-shrink');
        } else {
            navbar.classList.remove('navbar-shrink');
        }
    });
}

// Back to top button
function initBackToTop() {
    const backToTop = document.getElementById('backToTop');
    if (!backToTop) return;
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            backToTop.classList.add('show');
        } else {
            backToTop.classList.remove('show');
        }
    });
    
    backToTop.addEventListener('click', function(e) {
        e.preventDefault();
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// Smooth scrolling for anchor links
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Form enhancements
function initFormEnhancements() {
    // Add floating label effect
    const formControls = document.querySelectorAll('.form-control');
    formControls.forEach(input => {
        if (input.value) {
            input.classList.add('filled');
        }
        
        input.addEventListener('blur', function() {
            if (this.value) {
                this.classList.add('filled');
            } else {
                this.classList.remove('filled');
            }
        });
    });
}

// Utility functions
function showLoading(show = true) {
    const spinner = document.querySelector('.spinner-wrapper');
    if (spinner) {
        spinner.style.display = show ? 'flex' : 'none';
    }
}

function showAlert(message, type = 'info') {
    if (typeof Swal !== 'undefined') {
        Swal.fire({
            toast: true,
            position: 'top-end',
            icon: type,
            title: message,
            showConfirmButton: false,
            timer: 3000,
            timerProgressBar: true
        });
    } else {
        alert(message);
    }
}

// File upload utilities
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateImageFile(file, maxSize = 10 * 1024 * 1024) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
    
    if (!allowedTypes.includes(file.type)) {
        showAlert('Please select a valid image file (JPEG, PNG, BMP)', 'error');
        return false;
    }
    
    if (file.size > maxSize) {
        showAlert('File size too large. Please select an image smaller than 10MB.', 'error');
        return false;
    }
    
    return true;
}

// Animation utilities
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = Math.floor(progress * (end - start) + start);
        element.textContent = current;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Initialize counter animations when elements come into view
const observerOptions = {
    threshold: 0.7,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const counters = entry.target.querySelectorAll('[data-counter]');
            counters.forEach(counter => {
                const value = parseInt(counter.dataset.counter);
                if (value) {
                    animateValue(counter, 0, value, 2000);
                }
            });
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe elements with counters
document.querySelectorAll('.stat-box, .stat-card').forEach(el => {
    observer.observe(el);
});

// Export for global use
window.OralScanAI = {
    showLoading,
    showAlert,
    formatFileSize,
    validateImageFile,
    animateValue
};