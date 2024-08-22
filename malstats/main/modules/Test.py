import sys
import time

from main.modules.Model import Model
from main.modules.GeneralData import GeneralData
import tensorflow as tf
from pathlib import Path
from UserDB import UserDB
from multiprocessing import freeze_support
import polars as pl
from AnimeDB import AnimeDB
from main.modules.general_utils import find_duplicates, timeit, terminate_program
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
from Model2 import ModelCreator, ModelParams, Model, UserScoresPredictor, ModelTester
from main.modules.MAL_utils import MALUtils


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


# @timeit
# def find_nearest_neighbors(query_scores, index, num_neighbors):
#     """
#     Finds the nearest neighbors for a given query using the Annoy index.
#
#     :param query_scores: The numpy array of scores for which to find neighbors.
#     :param index: The AnnoyIndex object.
#     :param num_neighbors: The number of nearest neighbors to find.
#     :return: A list of tuples (index, distance) for the nearest neighbors.
#     """
#     return index.get_nns_by_vector(query_scores, num_neighbors, include_distances=True)


def main():
    # # freeze_support()
    # anime_db = AnimeDB()
    # anime_db.generate_anime_DB(update=True)
    # test = MALUtils.get_recent_ptw_data("Jun 19, 11:59 PM", "Jun 20, 11:59 PM")
    # anime_db = AnimeDB()
    # anime_db.generate_anime_DB(update=True)

    # anime_db = AnimeDB()
    # anime_db.generate_anime_DB(update=True)
    # print(5)
    # tags = Tags()
    # test = tags.tags_per_category
    # ModelTester().test_all(starting_model_index=103)
    # model_params = ModelParams(layer_count=3, layer_sizes=[1024, 1024, 1024], algo="RMSprop",
    #                            lr=0.001, reg_rate=0, loss="mean_absolute_error")
    # model_creator = ModelCreator(model_params=model_params)
    # model = model_creator.create()
    # model.train(epochs=23)
    #
    # model_params = ModelParams(layer_count=3, layer_sizes=[6144, 6144, 6144], algo="RMSprop",
    #                            lr=0.001, reg_rate=0, loss="mean_absolute_error")
    # model_creator = ModelCreator(model_params=model_params)
    # model = model_creator.create()
    # model.train(epochs=23)
    # ModelTester().test_all(starting_model_index=107)
    #
    # # model_params = ModelParams(layer_count=3, layer_sizes=[1024, 1024, 1024], algo="RMSprop",
    # #                            lr=0.001, reg_rate=0, loss="mean_absolute_error")
    # # model_creator = ModelCreator(model_params=model_params)
    # # model = model_creator.create()
    # # model.train(epochs=50)
    #
    # model_params = ModelParams(layer_count=3, layer_sizes=[1280, 1280, 1280], algo="RMSprop",
    #                            lr=0.001, reg_rate=0, loss="mean_absolute_error")
    # model_creator = ModelCreator(model_params=model_params)
    # model = model_creator.create()
    # model.train(epochs=50)
    # test = AnimeDB().generate_anime_DB(update=True)
    # test = find_max_affinity("BaronBrixius")
    # model = Model(tf.keras.models.load_model(
    #     main_model_path.parent / "Main_prediction_model102.h5"))
    # # test = model.predict_scores("BaronBrixius", "MAL")
    # # # model_params = ModelParams.random_init()
    # predictor = UserScoresPredictor(user_name="abyssbel", model=model)
    # test = predictor.predict_scores()

    # # print(5)
    # # user_db = UserDB()
    # # user_db.split_scores_dict()
    # # test = find_max_affinity("BaronBrixius")
    #
    # test = SeasonalStats2(username="BaronBrixius", site="MAL").full_stats.to_dict()
    test2 = SeasonalStats2(username="AmethystItalian", site="MAL").full_stats.to_dict()

    print(5)
    # while True:
    #     model = ModelCreator(random_init=True).create()
    #     model.train()
    # terminate_program()



    # aff_db = AffinityDB()
    # aff_db.create_minor_parts((2, 10))
    # print(5)
    # user_db = UserDB(continue_filling=True).df
    # model = Model()
    # model.train_model_ensemble()
    # stats = SeasonalStats2("BaronBrixius", "MAL").full_stats.to_dict()
    # no_seq_stats = SeasonalStats2("BaronBrixius", "MAL", no_sequels=True).full_stats.to_dict()
    # stats = SeasonalStats2("BaronBrixius", "MAL")
    # no_seq_stats = SeasonalStats2("BaronBrixius", "MAL", no_sequels=True)
    # stats = SeasonalStats2("Voltabolt", "Anilist").full_stats.to_dict()
    # no_seq_stats = SeasonalStats2("Voltabolt", "Anilist", no_sequels=True).full_stats.to_dict()

    # test = stats.full_stats
    # test2 = no_seq_stats.full_stats
    # test3 = test.to_dict()

    # fate_graph = load_pickled_file(data_path / "lupin_graph.pickle")
    # fate_graph.split()
    # test = load_pickled_file(data_path / "unsplit_graphs2.pickle")
    # # given_graph = test.graphs['Shiguang Dailiren']
    # # k = given_graph.split()
    #
    # tags = Tags()
    # test1 = tags.entry_tags_dict
    # test2 = tags.entry_tags_dict_nls
    # test3 = tags.show_tags_dict
    # test4 = tags.show_tags_dict_nls
    # test.split_graphs()
    # k = test.split()
    # graphs = Graphs2()
    # # test = graphs.all_graphs
    #
    # test2 = graphs.all_graphs
    # test = graphs.all_graphs_no_low_scores
    # # test = model.predict_scores("Voltabolt", site="Anilist")
    # test2 = find_max_affinity("BaronBrixius", site="MAL")
    # test = stats.get_user_seasonal_stats2()
    # test2 = stats.get_user_seasonal_stats("Voltabolt", "Anilist")

    # test2 = stats.get_user_seasonal_stats2("BaronBrixius")
    # print(5)
    # test_list = AnilistHandler("Voltabolt").anime_list.list
    # # test_formatter = ListFormatter(test_list, stats_to_get=["score", "list_status", "num_watched"])
    # # test_formatted_list = test_formatter.formatted_list
    #
    # test_list2 = MALListHandler("BaronBrixius").anime_list.list
    # print(5)
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
    # model = Model(model_filename=models_path / f"T1-1-50-{model_filename_suffix}.h5",
    #               layers=4, neuron_start=1536, neuron_lower=[False, True, False], algo="RMSprop",
    #               lr=0.003, loss='mean_absolute_error', epochs=50)
    # model.train()
    # print(5)
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