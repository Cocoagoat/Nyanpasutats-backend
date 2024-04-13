import sys
from main.modules.Model import Model
from main.modules.GeneralData import GeneralData
from pathlib import Path
from UserDB import UserDB
from multiprocessing import freeze_support
import polars as pl
from AnimeDB import AnimeDB
from main.modules.general_utils import find_duplicates, timeit
from main.modules.filenames import *
from main.modules.Tags import Tags
from main.models import TaskQueue
from main.tasks import get_user_seasonal_stats_task
from main.modules.general_utils import save_pickled_file, list_to_uint8_array, load_pickled_file
from main.modules.AffinityDB import AffinityDB
import random
from main.modules.AffinityFinder import find_max_affinity
from main.modules.SeasonalStats import SeasonalStats
from main.modules.SeasonalStats2 import SeasonalStats2
from animisc.celery import app
from main.modules.AnimeListHandler import MALListHandler, AnilistHandler
from main.modules.AnimeListFormatter import ListFormatter
from annoy import AnnoyIndex
from Graphs2 import Graphs2



# def train_model_ensemble():
#     i = 1
#     while True:
#         layers = random.choice([4, 5, 6, 7])
#         neuron_start = random.choice([256, 512, 1024])
#         neuron_lower = [bool(round(random.randint(0, 1))) for _ in range(5)]
#         algo = random.choice(["Adam", "Adagrad", "RMSprop", "SGD"])
#         lr = random.choice([0.01, 0.02, 0.04, 0.07])
#         loss = random.choice(["mean_squared_error", "mean_absolute_error"])
#         # epochs = random.choice([25, 40, 50])
#         # with_mean = bool(round(random.randint(0, 1)))
#         model = Model(model_filename=models_path / f"T4-{i}-50-RSDDPC.h5", layers=layers,
#                       neuron_start=neuron_start, neuron_lower=neuron_lower, algo=algo, lr=lr, loss=loss)
#         # add regularizer?
#         model.train(epochs=50)
#
#         i += 1

# @timeit
# def build_ann_index(scores_dict, n_trees=10):
#     """
#     Builds an Annoy index for the given score arrays.
#
#     :param score_arrays: A list of numpy arrays, where each array represents user scores.
#     :param n_trees: The number of trees for the Annoy index. Higher is more accurate, but slower to build.
#     :return: An AnnoyIndex object.
#     """
#     # k = score_arrays[0].shape[0]
#     k = len(scores_dict[list(scores_dict.keys())[0]])# Assuming all arrays are of the same length
#     t = AnnoyIndex(k, 'euclidean')  # Using Euclidean distance
#
#     # for i, scores in enumerate(score_arrays):
#     #     t.add_item(i, scores)
#
#     for i,(user_name, user_list) in enumerate(scores_dict.items()):
#         t.add_item(i, user_list)
#
#     t.build(n_trees)
#     return t


@timeit
def find_nearest_neighbors(query_scores, index, num_neighbors):
    """
    Finds the nearest neighbors for a given query using the Annoy index.

    :param query_scores: The numpy array of scores for which to find neighbors.
    :param index: The AnnoyIndex object.
    :param num_neighbors: The number of nearest neighbors to find.
    :return: A list of tuples (index, distance) for the nearest neighbors.
    """
    return index.get_nns_by_vector(query_scores, num_neighbors, include_distances=True)


def main():
    freeze_support()
    # user_db = UserDB()
    # scores_dict = user_db.scores_dict
    # t = build_ann_index(scores_dict)
    # my_list = MALListHandler("BaronBrixius").get_user_scores_list()
    # my_list = list_to_uint8_array(my_list)
    # test = find_nearest_neighbors(my_list, t, 50)
    # find_max_affinity("BaronBrixius")
    # aff_db = AffinityDB()
    # aff_db.create_minor_parts((3,10))
    # neuron_lower = [False, False, True, False, False, True]
    # model = Model(models_path / "T4-12-50-RSDDP.h5", layers=6, neuron_start=512, neuron_lower=neuron_lower, algo="Adam",
    #                lr=0.002,loss="mean_relative_error")
    # model = Model(models_path / "T4-19-50-RSDDP.h5")
    # aff_db = AffinityDB()
    # aff_db.create_minor_parts((4,10))
    # data = load_pickled_file(data_path / "general_data.pickle")
    # aff_db = AffinityDB()
    # aff_db.create_minor_parts((4,10))
    # data = aff_db.get_means_of_OG_affs()
    # save_pickled_file(data_path / "general_data.pickle", data)
    # tags = Tags()
    # test = tags.show_tags_dict
    # task_id = 'the-task-id'
    # task_result = app.AsyncResult(task_id)
    # print('Task status:', task_result.status)
    # print('Task result:', task_result.result)
    # task = get_user_seasonal_stats_task.delay("BaronBrixius")
    # test = TaskQueue.objects.all()
    # user_db = UserDB()
    # test = user_db.get_user_db_entry("BaronBrixius")
    # print(10)
    # current_dir = Path(__file__).parent.parent
    # test = find_max_affinity("RedInfinity")
    # print(5)
    # model = Model(model_filename = current_dir.parent / "MLmodels" / current_model_name)
    # test = SeasonalStats.get_user_seasonal_stats("BaronBrixius", "MAL")
    # stats = SeasonalStats2("BaronBrixius", "MAL").full_stats.to_dict()
    # no_seq_stats = SeasonalStats2("BaronBrixius", "MAL", no_sequels=True).full_stats.to_dict()
    # stats = SeasonalStats2("BaronBrixius", "MAL")
    # no_seq_stats = SeasonalStats2("BaronBrixius", "MAL", no_sequels=True)
    # stats = SeasonalStats2("Voltabolt", "Anilist").full_stats.to_dict()
    # no_seq_stats = SeasonalStats2("Voltabolt", "Anilist", no_sequels=True).full_stats.to_dict()

    # test = stats.full_stats
    # test2 = no_seq_stats.full_stats
    # test3 = test.to_dict()
    graphs = Graphs2()
    # test = graphs.all_graphs

    test2 = graphs.all_graphs
    test = graphs.all_graphs_no_low_scores
    # test = model.predict_scores("Voltabolt", site="Anilist")
    test2 = find_max_affinity("BaronBrixius", site="MAL")
    # test = stats.get_user_seasonal_stats2()
    # test2 = stats.get_user_seasonal_stats("Voltabolt", "Anilist")

    # test2 = stats.get_user_seasonal_stats2("BaronBrixius")
    # print(5)
    # test_list = AnilistHandler("Voltabolt").anime_list.list
    # # test_formatter = ListFormatter(test_list, stats_to_get=["score", "list_status", "num_watched"])
    # # test_formatted_list = test_formatter.formatted_list
    #
    # test_list2 = MALListHandler("BaronBrixius").anime_list.list
    print(5)
    # test_formatter2 = ListFormatter(test_list2)
    # test_formatted_list2 = test_formatter2.formatted_list
    #
    # test_list3 = MALListHandler("BaronBrixius").anime_list
    # test_formatter3 = ListFormatter(test_list2)
    # test_formatted_list3 = test_formatter2.formatted_list
    #
    # mal_handler = MALListHandler("BaronBrixius")
    # test_list = mal_handler.anime_list
    # test_formatter = MALListFormatter(anime_list=test_list, stats_to_get=["score", "status"])
    # anilist_handler = AnilistHandler("Voltabolt")
    # test = anilist_handler.get_user_scores_list()
    # print(5)
    # aff_db = AffinityDB()
    # model=Model()
    # model.predict_scores("BaronBrixius")
    # model.calculate_mean_pred_deviation()
    # model = Model(layers=3, neuron_start=1024,
    #               neuron_lower=[False, False], algo="RMSprop", lr=0.0005,loss='mean_absolute_error', epochs=50,
    #               reg=0.0005)
    # model.create_deviation_model()
    # model = Model(model_filename=models_path / f"T1-70-50-{model_filename_suffix}.h5", layers=3, neuron_start=1536,
    #               neuron_lower=[False, False], algo="RMSprop", lr=0.003, loss='mean_absolute_error', epochs=100)
    # model.train()
    # model = Model(model_filename=models_path / f"T1-61-50-{model_filename_suffix}.h5", layers=4, neuron_start=1024,
    #               neuron_lower=[False, False, False], algo="Adam", lr=0.003, loss='mean_absolute_error')
    # model.train(20)
    # model = Model(model_filename=models_path / f"T1-62-50-{model_filename_suffix}.h5", layers=4, neuron_start=1536,
    #               neuron_lower=[False, False, False], algo="RMSprop", lr=0.003, loss='mean_absolute_error')
    # model.train(20)
    # model = Model(model_filename=models_path / f"T1-63-50-{model_filename_suffix}.h5", layers=4, neuron_start=1536,
    #               neuron_lower=[False, False, False], algo="Adam", lr=0.003, loss='mean_absolute_error')
    # model.train(20)

    #def __init__(self, model_filename=None, batch_size=2048, user_name=None, with_mean=False, with_extra_doubles=False,
                 # layers=3, neuron_start=1024, neuron_lower=[False, False, False], algo="Adam", lr=0.002,
                 # loss='mean_absolute_error', ):

    # model.train_model_ensemble(starting_model_index=41)
    # find_max_affinity("BaronBrixius")
    # model = Model()
    # model.fetch_deviations()
    # model = Model()
    # model.calculate_mean_pred_deviation()
    # model.predict_scores("BaronBrixius")
    # model.test_models(starting_model_index=69)

    # model.train(50)
    # model.predict_scores("BaronBrixius")
    # print(5)