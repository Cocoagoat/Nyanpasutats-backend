from datetime import timedelta
from pathlib import Path
from django.apps import AppConfig
from celery.schedules import crontab
import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.set_visible_devices([], 'GPU')

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY")
MEDIA_URL = 'https://nps.moe/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'main/data/media')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_celery_results',
    'main.apps.MainConfig',
    'rest_framework',
    'corsheaders'
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/2'
CELERY_TIMEZONE = 'Europe/Helsinki'
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_EXPIRES = 3600


CELERY_BEAT_SCHEDULE = {
    'update_anime_db': {
        'task': 'main.tasks.daily_update',
        'schedule': crontab(hour="00", minute="52"),
    },
    'clean_up_folder': {
        'task': 'main.clean_up_folder',
        'schedule': crontab(hour="20", minute="15")
    }
}

ROOT_URLCONF = 'animisc.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'animisc.wsgi.application'

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://localhost:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
    },
}

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = CELERY_TIMEZONE

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Directory where your logs are stored
LOG_DIR = os.path.join(BASE_DIR, 'Logs')

# Ensure the directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'base': {
            'format': '[%(asctime)s] %(name)s:%(levelname)s in %(funcName)s at line %(lineno)d: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
        },
        'full': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'full.log'),
            'maxBytes': 1024*1024*4,
            'backupCount': 5,
            'formatter': 'base',
            'delay': True,  # Delays file opening until needed
        },
        'affinity': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'affinity.log'),
            'maxBytes': 1024*1024*5,
            'backupCount': 2,
            'formatter': 'base'
        },
        'recs': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'recs.log'),
            'maxBytes': 1024*1024*5,
            'backupCount': 2,
            'formatter': 'base'
        },
        'seasonal': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'seasonal.log'),
            'maxBytes': 1024*1024*5,
            'backupCount': 2,
            'formatter': 'base'
        },
        'alert': {
            'level': 'WARNING',
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_DIR, 'alert.log'),
            'formatter': 'base'
        },
    },
    'loggers': {
        'django': {
            'handlers': ['full', 'alert'],
            'level': 'INFO',
            'propagate': True,
        },
        'nyanpasutats': {
            'handlers': ['full', 'alert'],
            'level': 'INFO',
            'propagate': True,
        },
        # 'nyanpasutats.tasks': {
        #     'handlers': ['full', 'alert'],
        #     'level': 'INFO',
        #     'propagate': True,
        # },
        'nyanpasutats.affinity_logger': {
            'handlers': ['affinity'],
            'level': 'INFO',
            'propagate': True
        },
        'nyanpasutats.recs_logger': {
            'handlers': ['recs'],
            'level': 'INFO',
            'propagate': True
        },
        'nyanpasutats.seasonal_logger': {
            'handlers': ['seasonal'],
            'level': 'INFO',
            'propagate': True
        },
        'django.utils.autoreload': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}