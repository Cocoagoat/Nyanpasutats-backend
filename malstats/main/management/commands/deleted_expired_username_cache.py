from django.core.management.base import BaseCommand
from django.utils import timezone
from main.models import UsernameCache
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Delete expired username cache entries'

    def handle(self, *args, **kwargs):
        try:
            now = timezone.now()
            logger.info(f"Current time: {now}")
            expired_entries = UsernameCache.objects.filter(expires_at__lte=now)
            count = expired_entries.count()
            logger.info(f"Found {count} expired username cache entries")
            expired_entries.delete()
            self.stdout.write(self.style.SUCCESS(f'Deleted {count} expired username cache entries'))
            logger.info(f'Deleted {count} expired username cache entries')
        except Exception as e:
            logger.error(f"Error deleting expired username cache entries: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Error deleting expired username cache entries: {str(e)}"))