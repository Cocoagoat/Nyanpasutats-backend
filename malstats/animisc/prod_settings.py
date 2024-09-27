import os
from .base_settings import *

DEBUG = False

CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = [
    'https://malstats.com',
    'https://www.malstats.com',
]

ALLOWED_HOSTS = ['localhost', '127.0.0.1',
                 'nyanpasutats.moe', 'www.nyanpasutats.moe']

SECURE_HSTS_SECONDS = 3600
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

SECURE_CONTENT_TYPE_NOSNIFF = True


