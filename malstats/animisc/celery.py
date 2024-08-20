from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from kombu import Exchange, Queue
from celery.schedules import crontab


# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'animisc.settings')

app = Celery('animisc')

app.conf.task_queues = (
    Queue('affinity', Exchange('affinity', type='direct'), routing_key='affinity'),
    Queue('recs', Exchange('recs', type='direct'), routing_key='recs'),
    Queue('seasonal', Exchange('seasonal', type='direct'), routing_key='seasonal'),
)

app.conf.beat_schedule = {
    'delete-expired-username-cache-every-hour': {
        'task': 'your_app.tasks.delete_expired_username_cache',
        'schedule': crontab(minute=0, hour='*'),  # Every hour at minute 0
    },
}


app.conf.task_routes = {
    'main.tasks.get_user_affs_task': {'queue': 'affinity'},
    'main.tasks.get_user_recs_task': {'queue': 'recs'},
    'main.tasks.get_user_seasonal_stats_task': {'queue': 'seasonal'},
}

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()