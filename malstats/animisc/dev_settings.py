from .base_settings import *

DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1']

SECURE_HSTS_SECONDS = 3600
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

SESSION_COOKIE_SECURE = True
# CSRF_COOKIE_SECURE = True

# CORS_ALLOWED_ORIGINS = [
#     "http://localhost:3000",
# ]
#
# CSRF_TRUSTED_ORIGINS = [
#     "http://localhost:3000",
# ]
#
# CORS_ALLOW_CREDENTIALS = True
#
# CORS_ALLOW_HEADERS = [
#     'Content-Type',
#     'Authorization',
#     'X-CSRFToken',
# ]
#
# CORS_ALLOW_METHODS = [
#     'GET',
#     'POST',
#     'OPTIONS',
#     'PUT',
#     'DELETE',
# ]