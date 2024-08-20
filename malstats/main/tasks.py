import time

from .modules.Model2 import Model, UserScoresPredictor
from pathlib import Path
from django.core.cache import cache
from .modules.filenames import main_model_path
import tensorflow as tf
from .modules.SeasonalStats2 import SeasonalStats2
from .modules.filenames import current_model_name
from celery import shared_task, Task
from animisc.celery import app
from .modules.general_utils import redis_cache_wrapper
from .modules.SeasonalStats import SeasonalStats  # Assuming the logic resides here
from .modules.Errors import UserListFetchError
from .modules.AffinityFinder import find_max_affinity
from .models import TaskQueue
from django.core.management import call_command
import logging

logger = logging.getLogger(__name__)


current_dir = Path(__file__).parent
model = None
user_db = None
CACHE_TIMEOUT = 3600
# tf.config.set_visible_devices([], 'GPU')

view_logger = logging.getLogger('Nyanpasutats.view')


# def delete_task(task_id):
#     task = TaskQueue.objects.get(task_id=task_id)
#     task.delete()


class MyTask(Task):
    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """ This method runs after the task finishes regardless of its state. """
        super().after_return(status, retval, task_id, args, kwargs, einfo)
        if status == "SUCCESS":
            print("Task success")
        elif status == "FAILURE":
            print("Task failed")

        # Decrement the counter in Redis
        print("Before decrementing", cache.get('tasks_in_queue'))

        cache_info = cache.client.get_client().connection_pool.connection_kwargs
        print("Cache configuration (after_return):", cache_info)

        try:
            cache.decr('tasks_in_queue')
            print("Inside try")
        except ValueError:
            print("Error - key tasks_in_queue not found in cache")

        print("After decrementing", cache.get('tasks_in_queue'))
        time.sleep(0.03)

    def before_start(self, task_id, args, kwargs):
        try:
            cache.incr('tasks_in_queue')
            cache.incr('tasks_in_queue')
        except ValueError:
            cache.set('tasks_in_queue', 2)
        print("Key after incrementing", cache.get('tasks_in_queue'))
        return super().before_start(task_id, args, kwargs)

    # def apply_async(self, args=None, kwargs=None, **options):
    #     """ Increment the counter in Redis before running the task. """
    #     try:
    #         cache.incr('tasks_in_queue')
    #     except ValueError:
    #         cache.set('tasks_in_queue', 1)
    #     return super().apply_async(args, kwargs, **options)



# def on_task_success(result, *args, **kwargs):
#     def on_task_success(result, *args, **kwargs):
#         print("Task success")
#         print(result, *args, **kwargs)
#
#
# def on_task_failure(result, *args, **kwargs):
#     print("Task failed")
#     print(result, *args, **kwargs)

@shared_task(base=MyTask)
@redis_cache_wrapper(timeout=CACHE_TIMEOUT)
def get_user_seasonal_stats_task(username, site="MAL"):
    print("Entering seasonal task")
    try:
        stats = SeasonalStats2(username, site).full_stats.to_dict()
        no_seq_stats = SeasonalStats2(username, site, no_sequels=True).full_stats.to_dict()

        # seasonal_dict, seasonal_dict_no_sequels = SeasonalStats.get_user_seasonal_stats(username)
        result = {'Stats': stats, 'StatsNoSequels': no_seq_stats}
        # cache.set(cache_key, result, CACHE_TIMEOUT)
        # cache.set("test", "1", CACHE_TIMEOUT)
        # print("Cache is supposed to be set at this point")
        return result
    except UserListFetchError as e:
        # cache.set(cache_key, result, 60) # test this
        return {'error': e.message, 'status': e.status}
    except Exception as e:
        logging.error(f"An unexpected error has occurred. {e}")
        return {'error': "An unexpected error has occurred on our side. Please try again later.", 'status': 500}


@shared_task(base=MyTask)
@redis_cache_wrapper(timeout=CACHE_TIMEOUT)
def get_user_recs_task(username, site="MAL"):
    print("Entering recommendations task")
    global model
    if not model:
        print("Initializing model")
        model = Model(tf.keras.models.load_model(
            main_model_path.parent / "Main_prediction_model.h5"))
        # test = username

    scores_predictor = UserScoresPredictor(user_name=username,
                                           model=model, site=site,
                                           shows_to_take="all")
    try:
        predictions, predictions_sorted_by_diff, fav_tags, least_fav_tags = scores_predictor.predict_scores()
        print("After returning from predict_scores")
        # predictions = predictions.astype(float)
        return {'Recommendations': predictions,
                'RecommendationsSortedByDiff': predictions_sorted_by_diff,
                'FavTags': fav_tags,
                'LeastFavTags': least_fav_tags}
    except UserListFetchError as e:
        return {'error': e.message, 'status': e.status}
    except Exception as e:
        logging.error(f"An unexpected error has occurred. {e}")
        return {'error': "An unexpected error has occurred on our side. Please try again later.", 'status': 500}


@shared_task(base=MyTask)
@redis_cache_wrapper(timeout=CACHE_TIMEOUT)
def get_user_affs_task(username, site="MAL"):
    print("Entering affinities task")
    try:
        pos_affinities, neg_affinities = find_max_affinity(username, site)
        print("After returning from find_max_affinities")
        return {'PosAffs': pos_affinities, 'NegAffs': neg_affinities}
    except UserListFetchError as e:
        return {'error': e.message, 'status': e.status}
    except Exception as e:
        logging.error(f"An unexpected error has occurred. {e}")
        return {'error': "An unexpected error has occurred on our side. Please try again later.", 'status': 500}


@shared_task
def delete_expired_username_cache():
    try:
        call_command('delete_expired_username_cache')
    except Exception as e:
        logger.error(f"Error running delete_expired_username_cache task: {str(e)}")