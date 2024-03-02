from .modules.Model import Model
from pathlib import Path
from .modules.filenames import current_model_name
from celery import shared_task
from .modules.SeasonalStats import SeasonalStats  # Assuming the logic resides here
from .modules.Errors import UserListFetchError

current_dir = Path(__file__).parent
model = None


# This will still be problematic cause every process that gets recs will initialize its own model?
@shared_task
def get_user_seasonal_stats_task(username):
    try:
        seasonal_dict, seasonal_dict_no_sequels = SeasonalStats.get_user_seasonal_stats(username)
        # You might want to store these results in a database or cache with a unique key
        return {'Stats': seasonal_dict, 'StatsNoSequels': seasonal_dict_no_sequels}
    except UserListFetchError as e:
        # Handle the error accordingly, maybe setting a failure state in the database/cache
        # self.update_state(state='FAILURE', meta={'exc_message': str(e.message)})
        return {'error': e.message, 'status': e.status}


@shared_task
def get_user_recommendations(username):
    global model
    if not model:
        model = Model(model_filename=current_dir / "MLmodels" / current_model_name)
    try:
        predictions, predictions_sorted_by_diff = model.predict_scores(username, db_type=1)
        print("After returning from predict_scores")
        # predictions = predictions.astype(float)
        return {'Recommendations': predictions,
                'RecommendationsSortedByDiff': predictions_sorted_by_diff}
    except UserListFetchError as e:
        return {'error': e.message, 'status': e.status}