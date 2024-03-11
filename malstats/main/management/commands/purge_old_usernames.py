from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from main.models import UsernameCache


class Command(BaseCommand):
    help = 'Purge entries older than X days'

    def add_arguments(self, parser):
        # Optional argument to specify age of entries to purge
        parser.add_argument('--hours', type=int, default=2)

    def handle(self, *args, **options):
        hours = options['hours']
        cutoff_date = timezone.now() - timedelta(hours=hours)
        # Adjust the filter according to your timestamp field
        old_entries = UsernameCache.objects.filter(created_at__lt=cutoff_date)
        count = old_entries.count()
        old_entries.delete()
        self.stdout.write(f'Deleted {count} entries older than {hours} hours.')