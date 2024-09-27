"""
WSGI config for malstats project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'animisc.dev_settings')
# Should be defined in virtual environment's activate script as well,
# if there's a settings-related bug make sure that it's there


application = get_wsgi_application()
