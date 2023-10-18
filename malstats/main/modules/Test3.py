from modules.Model import Model
from pathlib import Path
from multiprocessing import freeze_support
import polars as pl
from modules.general_utils import find_duplicates
from modules.filenames import *
from modules.AffinityDB import AffinityDB
import random


def train_model_ensemble():
    i = 1
    while True:
        layers = random.choice([4, 5, 6, 7])
        neuron_start = random.choice([256, 512, 1024])
        neuron_lower = [bool(round(random.randint(0, 1))) for _ in range(5)]
        algo = random.choice(["Adam", "Adagrad", "RMSprop", "SGD"])
        lr = random.choice([0.01, 0.02, 0.04, 0.07])
        loss = random.choice(["mean_squared_error", "mean_absolute_error"])
        # epochs = random.choice([25, 40, 50])
        # with_mean = bool(round(random.randint(0, 1)))
        model = Model(model_filename=models_path / f"T4-{i}-50-RSDDPC.h5", layers=layers,
                      neuron_start=neuron_start, neuron_lower=neuron_lower, algo=algo, lr=lr, loss=loss)
        # add regularizer?
        model.train(epochs=50)

        i += 1


def main():
    freeze_support()
    # aff_db = AffinityDB()
    # aff_db.create_minor_parts((3,10))
    # neuron_lower = [False, False, True, False, False, True]
    # model = Model(models_path / "T4-12-50-RSDDP.h5", layers=6, neuron_start=512, neuron_lower=neuron_lower, algo="Adam",
    #                lr=0.002,loss="mean_relative_error")
    # model = Model(models_path / "T4-19-50-RSDDP.h5")
    model = Model()
    # model.train_model_ensemble()
    model.test_models()
    # model.train(50)
    # model.predict_scores("BaronBrixius")
    # print(5)