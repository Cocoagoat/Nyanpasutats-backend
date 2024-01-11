from main.modules.Model import Model
from main.modules.GeneralData import GeneralData
from pathlib import Path
from multiprocessing import freeze_support
import polars as pl
from main.modules.general_utils import find_duplicates
from main.modules.filenames import *
from main.modules.general_utils import save_pickled_file, load_pickled_file
from main.modules.AffinityDB import AffinityDB
import random
from main.modules.AffinityFinder import find_max_affinity
from main.modules.SeasonalStats import SeasonalStats


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


def main():
    freeze_support()
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

    stats = SeasonalStats(start_year=2017)
    test = stats.get_user_seasonal_stats("BaronBrixius")
    # test2 = stats.get_user_seasonal_stats2("BaronBrixius")
    print(5)
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