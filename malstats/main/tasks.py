from .modules.Model import Model
from pathlib import Path
from django.core.cache import cache
from .modules.filenames import current_model_name
from celery import shared_task, Task
from animisc.celery import app
from .modules.general_utils import redis_cache_wrapper
from .modules.SeasonalStats import SeasonalStats  # Assuming the logic resides here
from .modules.Errors import UserListFetchError
from .modules.AffinityFinder import find_max_affinity
from .models import TaskQueue
import logging


current_dir = Path(__file__).parent
model = None
user_db = None
CACHE_TIMEOUT = 3600

view_logger = logging.getLogger('Nyanpasutats.view')


# def delete_task(task_id):
#     task = TaskQueue.objects.get(task_id=task_id)
#     task.delete()


class MyTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        print("Task success")
        try:
            print("Attempting to delete task ", task_id)
            self.delete_task(task_id)
        except TaskQueue.DoesNotExist:
            print(f"Unable to delete task {task_id}")
            view_logger.error(f"Unable to delete task {task_id}")

    def on_task_failure(self, retval, task_id, args, kwargs):
        print("Task failed")
        try:
            self.delete_task(task_id)
        except TaskQueue.DoesNotExist:
            print(f"Unable to delete task {task_id}")
            view_logger.error(f"Unable to delete task {task_id}")

    @staticmethod
    def delete_task(task_id):
        print(f"Fetching task {task_id} to delete")
        task = TaskQueue.objects.get(task_id=task_id)
        print("Deleting task",task)
        task.delete()


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
def get_user_seasonal_stats_task(username):
    print("Entering seasonal task")
    try:
        seasonal_dict, seasonal_dict_no_sequels = SeasonalStats.get_user_seasonal_stats(username)
        result = {'Stats': seasonal_dict, 'StatsNoSequels': seasonal_dict_no_sequels}
        # cache.set(cache_key, result, CACHE_TIMEOUT)
        # cache.set("test", "1", CACHE_TIMEOUT)
        # print("Cache is supposed to be set at this point")
        return result
    except UserListFetchError as e:
        # cache.set(cache_key, result, 60) # test this
        return {'error': e.message, 'status': e.status}
    except Exception as e:
        if not e.status:
            e.status = 400
        # I know this is bad practice, but in case of an unexpected error in fetching
        # an individual user's stats, we do not want the site to crash.
        logging.error(f"An unexpected error has occurred. {e.status} {e.message}")
        return {'error': e.message, 'status': e.status}


@shared_task(base=MyTask)
@redis_cache_wrapper(timeout=CACHE_TIMEOUT)
def get_user_recs_task(username):
    print("Entering recommendations task")
    global model
    if not model:
        print("Initializing model")
        model = Model(model_filename=current_dir / "MLmodels" / current_model_name)
    try:
        predictions, predictions_sorted_by_diff, fav_tags, least_fav_tags = model.predict_scores(username, db_type=1)
        print("After returning from predict_scores")
        # predictions = predictions.astype(float)
        return {'Recommendations': predictions,
                'RecommendationsSortedByDiff': predictions_sorted_by_diff,
                'FavTags': fav_tags,
                'LeastFavTags': least_fav_tags}
    except UserListFetchError as e:
        return {'error': e.message, 'status': e.status}
    except Exception as e:
        if not e.status:
            e.status = 400
        logging.error(f"An unexpected error has occurred. {e.message}")
        return {'error': e.message, 'status': e.status}


@shared_task(base=MyTask)
@redis_cache_wrapper(timeout=CACHE_TIMEOUT)
def get_user_affs_task(username):
    print("Entering affinities task")
    try:
        pos_affinities, neg_affinities = find_max_affinity(username)
        print("After returning from find_max_affinities")
        return {'PosAffs': pos_affinities, 'NegAffs': neg_affinities}
    except UserListFetchError as e:
        return {'error': e.message, 'status': e.status}
    except Exception as e:
        if not e.status:
            e.status = 400
        logging.error(f"An unexpected error has occurred. {e.status} {e.message}")
        return {'error': e.message, 'status': e.status}
