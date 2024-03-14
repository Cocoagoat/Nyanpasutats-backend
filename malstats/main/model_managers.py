from django.db import models
from django.db.utils import OperationalError
import time
import logging

view_logger = logging.getLogger('Nyanpasutats.view')


class RetryManager(models.Manager):
    def get(self, *args, retries=50, delay=1, **kwargs):
        for attempt in range(retries):
            try:
                # print(f"Getting {args, kwargs}")
                return super().get(*args, **kwargs)
            except OperationalError:
                view_logger.error("Caught Operational Error")
                if attempt + 1 == retries:
                    raise
                time.sleep(delay)

    def filter(self, *args, retries=50, delay=1, **kwargs):
        for attempt in range(retries):
            try:
                # print(f"Filtering {args, kwargs}")
                return super().filter(*args, **kwargs)
            except OperationalError:
                view_logger.error("Caught Operational Error")
                if attempt + 1 == retries:
                    raise
                time.sleep(delay)

    def get_or_create(self, *args, retries=50, delay=1, **kwargs):
        for attempt in range(retries):
            try:
                # print(f"Getting_or_creating {args, kwargs}")
                return super().get_or_create(*args, **kwargs)
            except OperationalError:
                view_logger.error("Caught Operational Error")
                if attempt + 1 == retries:
                    raise
                time.sleep(delay)

    def create(self, *args, retries=50, delay=1, **kwargs):
        for attempt in range(retries):
            try:
                print(f"Creating {args, kwargs}")
                return super().create(*args, **kwargs)
            except OperationalError:
                view_logger.error("Caught Operational Error")
                if attempt + 1 == retries:
                    raise
                time.sleep(delay)