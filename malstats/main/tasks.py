from .modules.Model import Model, UserScoresPredictor
from django.core.cache import cache
import shutil
from .modules.Tags import Tags
import tensorflow as tf
from .modules.SeasonalStats import SeasonalStats
from .modules.AnimeDB import AnimeDB
from .modules.filenames import *
from celery import shared_task, Task
from .modules.general_utils import redis_cache_wrapper, determine_queue_cache_key, add_suffix_to_filename
from .modules.Errors import UserListFetchError
from .modules.AffinityFinder import find_max_affinity
from .modules.Model import add_image_urls_to_predictions
from celery.exceptions import TimeLimitExceeded
from django.db.utils import OperationalError
import logging
import time
import traceback

current_dir = Path(__file__).parent
model = None
user_db = None
CACHE_TIMEOUT = 3600*3
tf.config.set_visible_devices([], 'GPU')

logger = logging.getLogger('nyanpasutats')
aff_logger = logging.getLogger('nyanpasutats.affinity_logger')
recs_logger = logging.getLogger('nyanpasutats.recs_logger')
seasonal_logger = logging.getLogger('nyanpasutats.seasonal_logger')


class MainTask(Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_count = 0  # for future use
        self.queue_cache_key = determine_queue_cache_key(self.name, "task")
        # separate caches for each queue (affinity, recs, seasonal)

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """ This method runs after the task finishes regardless of its state. """
        super().after_return(status, retval, task_id, args, kwargs, einfo)
        try:
            cache.decr(self.queue_cache_key)
        except ValueError:
            cache.set(self.queue_cache_key, 1, timeout=3600)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} with args {args}, {kwargs} failed to execute. Error info : {exc}")

    def apply_async(self, args=None, kwargs=None, task_id=None, producer=None,
                    link=None, link_error=None, **options):
        time.sleep(0.5)  # Sleep a bit before starting task for queue to properly catch up

        try:
            cache.incr(self.queue_cache_key)
        except ValueError:
            cache.set(self.queue_cache_key, 2, timeout=3600)

        return super().apply_async(args=args, kwargs=kwargs, task_id=task_id,
                                   producer=producer, link=link, link_error=link_error,
                                   **options)


@shared_task(base=MainTask, autoretry_for=(OperationalError,), max_retries=10)
@redis_cache_wrapper(timeout=CACHE_TIMEOUT)
def get_user_seasonal_stats_task(username, site="MAL"):
    seasonal_logger.info(f"Entering seasonal task for {site} user {username}")
    try:
        stats = SeasonalStats(username, site).full_stats.to_dict()
        no_seq_stats = SeasonalStats(username, site, no_sequels=True).full_stats.to_dict()
        result = {'Stats': stats, 'StatsNoSequels': no_seq_stats}
        seasonal_logger.info(f"Successfully fetched seasonal stats for {site} user {username}")
        return result
    except UserListFetchError as e:
        return {'error': e.message, 'status': e.status}

    except Exception as e:
        seasonal_logger.error(f"An unexpected error has occurred while fetching seasonal stats. {str(e)}")
        return {'error': "An unexpected error has occurred on our side. Please try again later.", 'status': 500}


@shared_task(base=MainTask)
@redis_cache_wrapper(timeout=CACHE_TIMEOUT)
def get_user_recs_task(username, site="MAL"):
    recs_logger.info(f"Entering recommendations task for {site} user {username}")
    global model
    if not model:
        recs_logger.info("Initializing model")
        model = Model(tf.keras.models.load_model(
            main_model_path.parent / "Main_prediction_model.h5"))

    try:
        scores_predictor = UserScoresPredictor(user_name=username,
                                               model=model, site=site,
                                               shows_to_take="all")
        predictions, predictions_sorted_by_diff, fav_tags, least_fav_tags = scores_predictor.predict_scores()
        predictions = add_image_urls_to_predictions(predictions)
        predictions_sorted_by_diff = add_image_urls_to_predictions(predictions_sorted_by_diff)
        print(f"Predictions look like {predictions[0]}")
        recs_logger.info(f"Predictions look like {predictions[0]}")
        recs_logger.info(f"Successfully fetched predictions for {site} user {username}")

        return {'Recommendations': predictions,
                'RecommendationsSortedByDiff': predictions_sorted_by_diff,
                'FavTags': fav_tags,
                'LeastFavTags': least_fav_tags}
    except UserListFetchError as e:
        return {'error': e.message, 'status': e.status}
    except Exception as e:
        recs_logger.error(f"An unexpected error has occurred while fetching recommendations. {str(e)}")
        return {'error': "An unexpected error has occurred on our side. Please try again later.", 'status': 500}


@shared_task(base=MainTask)
@redis_cache_wrapper(timeout=CACHE_TIMEOUT)
def get_user_affs_task(username, site="MAL"):
    aff_logger.info(f"Entering affinities task for {site} user {username}")
    try:
        pos_affinities, neg_affinities = find_max_affinity(username, site)
        aff_logger.info(f"Successfully fetched affinities for {site} user {username}")
        return {'PosAffs': pos_affinities, 'NegAffs': neg_affinities}
    except UserListFetchError as e:
        return {'error': e.message, 'status': e.status}
    except Exception as e:
        aff_logger.error(f"An unexpected error has occurred while fetching affinities. {str(e)}")
        return {'error': "An unexpected error has occurred on our side. Please try again later.", 'status': 500}


@shared_task
def update_anime_db():
    try:
        AnimeDB().generate_anime_DB(update=True)
    except Exception as e:
        logger.error(f"An unexpected error has occurred while updating the anime database. {str(e)}")


@shared_task(bind=True, autoretry_for=(TimeLimitExceeded,), retry_kwargs={'max_retries': 10, 'countdown': 60},
             soft_time_limit=6600, default_retry_delay=60, time_limit=7200, name='main.tasks.daily_update')
def daily_update(self):
    try:
        daily_backup()
        AnimeDB(anime_database_updated_name).generate_anime_DB(update=True)
        Tags().update_tags()
    except Exception as e:
        full_stack_trace = traceback.format_exc()  # Capture the full traceback as a string
        logger.error(f"An unexpected error has occurred during the daily update. {str(e)}")
        logger.error(full_stack_trace)
        self.retry()


def daily_backup():
    filenames_to_backup = [entry_tags_updated_filename,
                           entry_tags_nls_updated_filename, shows_tags_updated_filename,
                           shows_tags_nls_updated_filename, graphs_dict_updated_filename,
                           graphs_dict_nls_updated_filename]

    # We actually use AnimeDB's "prev" for updating the rest of the objects, so in case the update of
    # AnimeDB goes wrong we don't want to lose the "prev". The rest of the prevs are just for backup in
    # case something goes wrong during updating one of them.
    shutil.copy(anime_database_updated_name, add_suffix_to_filename(anime_database_updated_name, "prev_before_update"))
    for filename in filenames_to_backup:
        try:
            shutil.copy(filename, add_suffix_to_filename(filename, "prev"))
        except FileNotFoundError:
            continue  # First time updating, -U files haven't been created yet
