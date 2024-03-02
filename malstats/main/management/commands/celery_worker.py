from django.core.management.base import BaseCommand
from django.utils import autoreload
import shlex
import subprocess


def restart_celery(concurrency):
    cmd = f'pkill -f "celery worker"'
    subprocess.call(shlex.split(cmd))
    cmd = f'celery -A animisc worker --loglevel=info --concurrency={concurrency}'
    print(shlex.split(cmd))
    subprocess.call(shlex.split(cmd))


class Command(BaseCommand):
    help = "Restarts the Celery worker with the specified concurrency"

    def add_arguments(self, parser):
        # Named (optional) argument
        parser.add_argument('concurrency', nargs='?', type=int, default=4, help='The number of concurrent workers')

    def handle(self, *args, **options):
        concurrency = options['concurrency']
        print(f'Starting celery workerr with autoreload and concurrency {concurrency}...')
        autoreload.run_with_reloader(lambda: restart_celery(concurrency))