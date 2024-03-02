import pandas as pd
from .filenames import *
import sys
from .general_utils import *
from .MAL_utils import *
from tensorflow import keras
import polars as pl
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1, l2
import matplotlib.pyplot as plt
import os
import re
import time
import numpy as np
import tensorflow as tf
from .UserDB import UserDB
from .Tags import Tags
from .AnimeDB import AnimeDB
from .AffinityDB import AffDBEntryCreator, GeneralData, User, AffinityDB, UserAffinityCalculator


class Model:

    def __init__(self, model_filename= models_path / current_model_name, batch_size=2048, user_name=None, with_mean=False, with_extra_doubles=False,
                 layers=3, neuron_start=1024, neuron_lower=[False, False], neuron_double=[False,False], algo="Adam", lr=0.002,
                 loss='mean_absolute_error', epochs=50, reg=0.001):

        tf.config.set_visible_devices([], 'GPU')
        self.model_filename = model_filename
        self.model = tf.keras.models.load_model(models_path / self.model_filename)
        # self.model_type = model_type
        self.user_name = user_name
        self.batch_size = batch_size
        self.with_mean = with_mean
        self.with_extra_doubles = with_extra_doubles

        self.layers = layers
        self.neuron_start = neuron_start
        self.neuron_lower = neuron_lower
        self.neuron_double = neuron_double
        self.algo = algo
        self.lr = lr
        self.loss = loss
        self.tags = Tags()
        self.user_db = UserDB()
        self.aff_db = AffinityDB()
        self.anime_db = AnimeDB()
        self.epochs = epochs
        self.reg = reg
        try:
            self.data = load_pickled_file(data_path / "general_data.pickle")
        except FileNotFoundError:
            self.data = GeneralData().generate_data()

    def repeat_layers(self):
        repeated_layers = []
        neuron_count = self.neuron_start
        for i in range(self.layers):
            print(neuron_count)
            repeated_layers.extend([
                tf.keras.layers.Dense(neuron_count, kernel_regularizer=l1(self.reg)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.PReLU()
            ])
            if i == self.layers-1:
                break
            neuron_count = neuron_count * (1/2)**self.neuron_lower[i]
            neuron_count = neuron_count * 2**self.neuron_double[i]

        return repeated_layers

    def choose_optimizer(self):
        if self.algo == 'Adam':
            return tf.keras.optimizers.Adam(learning_rate=self.lr)
        elif self.algo == 'SGD':
            return tf.keras.optimizers.SGD(learning_rate=self.lr)
        elif self.algo == 'Adagrad':
            return tf.keras.optimizers.Adagrad(learning_rate=self.lr)
        elif self.algo == 'RMSprop':
            return tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        else:
            raise ValueError("Unknown optimization algorithm")

    def _create_model(self, num_features):
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Dense(self.neuron_start, input_shape=(num_features,)),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.Dense(512),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.Dense(512),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.PReLU(),
        #     # tf.keras.layers.Dense(1024),
        #     # tf.keras.layers.BatchNormalization(),
        #     # tf.keras.layers.PReLU(),
        #     # tf.keras.layers.Dense(1024),
        #     # tf.keras.layers.BatchNormalization(),
        #     # tf.keras.layers.PReLU(),
        #     tf.keras.layers.Dense(1)
        # ])

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_features,)),
            *self.repeat_layers(),
            tf.keras.layers.Dense(1)
        ])
        return model

    def train_model_ensemble(self, starting_model_index=1):
        i = starting_model_index
        while True:
            self.layers = random.choice([3, 4, 5])

            self.neuron_start = random.choice([1024, 1536])
            self.neuron_lower = [bool(round(random.randint(0, 1))) for _ in range(self.layers)]
            self.algo = random.choice(["Adam", "RMSprop", "SGD"])
            self.lr = random.choice([0.01, 0.02, 0.03, 0.05])
            self.loss = random.choice(["mean_squared_error", "mean_absolute_error"])
            self.epochs = random.choice([10,20,30,50,75,100,150])
            # self.model_type = model_type
            self.model_filename = models_path / f"T1-{i}-50-{model_filename_suffix}.h5"
            # with_mean = bool(round(random.randint(0, 1)))
            # model = Model(models_path / f"T4-{i}-50-RSDDP.h5", layers=layers,
            #               neuron_start=neuron_start, neuron_lower=neuron_lower, algo=algo, lr=lr, loss=loss)
            # add regularizer?
            self.train()
            i += 1

    def train(self):

        def interleave(files):
            #  thanks chatgpt very cool
            return files.interleave(lambda file_path: tf.data.Dataset.from_generator(
                self._generate_batch, args=[file_path], output_signature=(
                    tf.TensorSpec(shape=(self.batch_size, num_features), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32))
            ), cycle_length=1)

        def get_interleaved_training_data():
            file_amount = AffinityDB.count_major_parts()
            if not self.with_mean:
                file_paths = [str(aff_db_path / f"{aff_db_filename}-P{_}-N-{model_filename_suffix}.parquet") for _ in
                              range(file_amount)]
            else:  # tf is this? remove later, RS doesnt exist anymore
                file_paths = [str(aff_db_path / f"{aff_db_filename}-P{_}-N-RS.parquet") for _ in range(file_amount)]

            files = tf.data.Dataset.list_files(file_paths)

            train_files = files.take(file_amount - 1)
            test_files = files.skip(file_amount - 1)

            train_dataset = interleave(train_files)
            test_dataset = interleave(test_files)

            return train_dataset, test_dataset

        # def create_model_skeleton(num_features):
        #     model = self._create_model(num_features)
        #     optimizer = self.choose_optimizer()
        #     model.compile(optimizer=optimizer,
        #                   loss=self.loss,
        #                   metrics=['mean_absolute_error'])
        #     return model

        if os.path.exists(aff_db_path / f"{aff_db_filename}-P1-N.parquet"):
            aff_db = pq.ParquetFile(aff_db_path / f"{aff_db_filename}-P1-N.parquet")
        else:
            aff_db = AffinityDB()
            if not os.path.exists(aff_db_path / f"{aff_db_filename}-P1.parquet"):
                print("Creating Affinity DB")
                aff_db.create()
            print("Normalizing Affinity DB")
            aff_db.normalize()
            aff_db = pq.ParquetFile(aff_db_path / f"{aff_db_filename}-P1-N.parquet")

        num_features = len(aff_db.schema.names) - 1
        if not self.with_mean:
            # num_features = 98
            # num_features -= 1
            num_features = 95 # make a dict of type : amount of features
            if not os.path.exists(aff_db_path / f"{aff_db_filename}-P1-N-{model_filename_suffix}.parquet"):
                AffinityDB.remove_show_score()
        # if not self.with_extra_doubles:
        #     num_features -= 120
        #     if not os.path.exists(f"Partials\\{aff_db_filename}-P1-N-RSD.parquet"):
        #         AffinityDB.remove_extra_doubles()

        print("Starting model creation")
        print(f"{self.epochs} {self.lr}")

        # model = self._create_model(num_features)

        # optimizer = self.choose_optimizer()

        # # model.compile(optimizer=optimizer,
        #               loss=self.loss,
        #               metrics=['mean_absolute_error'])

        model = self.create_model_skeleton(num_features)

        # file_amount = AffinityDB.count_major_parts()
        # if not self.with_mean:
        #     file_paths = [str(aff_db_path / f"{aff_db_filename}-P{_}-N-{model_filename_suffix}.parquet") for _ in range(file_amount)]
        # else:  # tf is this? remove later, RS doesnt exist anymore
        #     file_paths = [str(aff_db_path / f"{aff_db_filename}-P{_}-N-RS.parquet") for _ in range(file_amount)]
        #
        # files = tf.data.Dataset.list_files(file_paths)
        #
        # train_files = files.take(file_amount - 1)
        # test_files = files.skip(file_amount - 1)
        #
        # train_dataset = interleave(train_files)
        # test_dataset = interleave(test_files)
        train_dataset, test_dataset = get_interleaved_training_data()
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        print("Beginning training")

        model.fit(
            train_dataset,
            epochs=self.epochs,  # Number of times the entire dataset is passed forward and backward through the neural network
            validation_data=test_dataset,
            verbose=1,  # To view the training progress for each epoch
            callbacks=[callback]
        )
        model.save(self.model_filename)

        self.create_deviation_model()

        # deviation_model = self.create_model_skeleton(num_features)
        # deviation_training_dataset = pd.read_parquet(aff_db_path / "AffinityDB-P1-N.parquet")
        # # The training dataset for the deviation model will basically be only 1 part of the 20+ used to train
        # #
        # user_scores = list(deviation_training_dataset['User Score'])
        #
        # # user_shows_df = user_shows_df_with_name.drop('Show Name', axis=1)
        # # normalized_df = self.aff_db.normalize_aff_df(user_shows_df, for_predict=True)
        # # normalized_df = normalized_df.fillna(0)
        # #
        # # normalized_df = normalized_df.drop('Show Score', axis=1)
        # normalized_dataset = AffinityDB.filter_df_for_model(deviation_training_dataset)
        # features = normalized_dataset.iloc[:, :-1].values
        # predictions = self.model.predict(features)
        # deviations = user_scores - predictions
        # X_train, X_test, y_train, y_test = train_test_split(predictions, deviations, test_size=0.1, random_state=42)
        # deviation_model.fit(
        #     x=normalized_dataset,
        #     y=predictions,
        #     epochs=self.epochs,
        #     validation_data=(X_test, y_test),
        #     batch_size=512
        # )
        #
        # model_name = self.model_filename.__str__().split("\\")[-1].split(".")[0] + "-Dev.h5"
        # # The above line just gets the filename (T1-4-50-RSDD, for example) from the full path which
        # # is stored in self.model_filename. <--- Rename later
        # deviation_model.save(models_path / model_name)

    def create_model_skeleton(self, num_features):
        model = self._create_model(num_features)
        optimizer = self.choose_optimizer()
        model.compile(optimizer=optimizer,
                      loss=self.loss,
                      metrics=['mean_absolute_error'])
        return model

    def _generate_batch(self, file_path_tensor):
        # Read a single parquet file using PyArrow
        file_path = file_path_tensor.decode('utf-8')
        df = pd.read_parquet(file_path)
        labels = df['User Score']
        features = df.drop(columns=['User Score'])
        total_rows = len(df)

        # Iterate through the DataFrame in chunks of size batch_size
        for i in range(0, total_rows - self.batch_size, self.batch_size):
            batch_features = features.iloc[i: i + self.batch_size]
            batch_labels = labels.iloc[i: i + self.batch_size]
            yield batch_features, batch_labels

    def create_deviation_model(self):
        deviation_training_dataset = pd.read_parquet(aff_db_path / "AffinityDB-P1-N.parquet")
        # The training dataset for the deviation model will basically be only 1 part of the 20+ used to train
        #
        user_scores = list(deviation_training_dataset['User Score'])

        # user_shows_df = user_shows_df_with_name.drop('Show Name', axis=1)
        # normalized_df = self.aff_db.normalize_aff_df(user_shows_df, for_predict=True)
        # normalized_df = normalized_df.fillna(0)
        #
        # normalized_df = normalized_df.drop('Show Score', axis=1)
        normalized_dataset = AffinityDB.filter_df_for_model(deviation_training_dataset)
        features = normalized_dataset.values
        num_features = normalized_dataset.shape[1]
        deviation_model = self.create_model_skeleton(num_features)
        predictions = self.model.predict(features)
        predictions = [x[0] for x in predictions]
        deviations = [(prediction - user_score) for prediction, user_score in list(zip(predictions, user_scores))]
        # sample_weights = [0.1*deviation + 0.1 for deviation in deviations]
        # predictions = (predictions - np.mean(predictions)) / np.std(predictions)
        # normalized_dataset['Predictions'] = predictions
        X_train, X_test, y_train, y_test = train_test_split(normalized_dataset, deviations, test_size=0.1, random_state=42)
        sample_weights = [0.01 * deviation + 0.1 for deviation in y_train]
        deviation_model.fit(
            x=np.float32(X_train),
            y=np.float32(y_train),
            epochs=self.epochs,
            sample_weight=np.array(sample_weights),
            validation_data=(np.float32(X_test), np.float32(y_test)),
            batch_size=512
        )

        model_name = self.model_filename.__str__().split("\\")[-1].split(".")[0] + "-Dev.h5"
        # The above line just gets the filename (T1-4-50-RSDD, for example) from the full path which
        # is stored in self.model_filename. <--- Rename later
        deviation_model.save(models_path / model_name)

    def test_models(self, starting_model_index=1):
        def adjust_predictions(preds):
            for p in preds:
                p['ScoreDiff'] = p['PredictedScore'] - p['MALScore']
            preds = sorted(preds, reverse=True, key = lambda x: x['ScoreDiff'])
            return preds

        def extract_array_value_from_file(filename, index=4, user_num=1):  # index=4 for the 5th value (0-based indexing)
            users_passed = 0
            with open(data_path / "ModelTests" / filename, "r") as file:
                for line in file:
                    if line.startswith('Errors :'):
                        users_passed += 1
                        if users_passed == user_num:
                            while True:
                                # The next line should contain the array
                                array_line = next(file).strip()
                                # Assuming the format of the array is consistent as shown
                                array_str = array_line.strip('[]')
                                array_values = array_str.split()
                                try:
                                    # Convert to float and return the desired value
                                    return float(array_values[index])
                                except (IndexError, ValueError):
                                    index -= len(array_values)
                                    # Handle cases where the index is out of bounds or conversion fails
                                    continue
                return None  # Return None if the pattern is not found

        shows_to_take = "watched+related"

        with open(data_path / "ModelTests" / "ModelTestUsernames.txt", 'r') as f:
            test_usernames = f.readlines()

        i = starting_model_index
        # Write the model details into each test file first (one file per model)
        while True:
            with open(data_path / "ModelTests" / f"ModelTest{i}-DX.txt", "w") as f:
                self.model_filename = models_path / f"T1-{i}-50-{model_filename_suffix}.h5"
                try:
                    model = tf.keras.models.load_model(models_path / self.model_filename)
                except (OSError, FileNotFoundError):
                    break
                f.write(f"Optimizer : {model.optimizer._name}\n")
                f.write(f"Loss : {model.loss}")
                f.write(f"Epochs : {self.epochs}")
                f.write(f"Learning rate : {self.lr}")
                f.write("Layers :\n")
                for layer in model.layers:
                    try:
                        f.write(f"{str(layer.units)}\n")
                    except AttributeError:
                        continue
            i += 1

        # Then grab all the test usernames and write results into each test file
        test_usernames = [username.strip() for username in test_usernames]
        for username in test_usernames:
            try:
                user_row = self.user_db.get_user_db_entry(username)
            except UserListFetchError as e:
                print(f"Failed to fetch {e.username}'s list")
                continue

            user = User(username, scores=user_row.select(self.data.relevant_shows).row(0, named=True),
                        scored_amount=user_row["Scored Shows"][0])
            user_shows_df_with_name, normalized_df = self.get_user_db(user, shows_to_take="all",
                                                                      with_mean=False)
            i = starting_model_index
            while True:
                with open(data_path / "ModelTests" / f"ModelTest{i}-DX.txt", "a", encoding='utf-8') as f:
                    self.model_filename = models_path / f"T1-{i}-50-{model_filename_suffix}.h5"
                    try:
                        model = tf.keras.models.load_model(self.model_filename) # remove the models_path?
                    except (OSError, FileNotFoundError):
                        break
                    # print(f"Currently open file index is {i}")
                    print(f"Fetching {self.model_filename} predictions for {username} ")
                    predictions, predictions_no_watched = self.fetch_predictions(normalized_df, user_shows_df_with_name,
                                                                 model, user, user_row, shows_to_take)
                    errors = self.calculate_error(predictions, user.mean_of_watched)
                    f.write(f"{username}\n")
                    # f.write(f"Model {i}")
                    f.write(f"Errors : \n{errors}\n")
                    # f.write(errors)
                    # f.write("Predictions: ", predictions)
                    f.write("Predictions\n")

                    for prediction in predictions[0:50]:
                        f.write(f"{prediction}\n")

                    predictions = adjust_predictions(predictions)
                    f.write("-------------------------------------------------------")

                    for prediction in predictions[0:50]:
                        f.write(f"{prediction}\n")

                i += 1

        model_count = i - 1
        # Finally, create the summary file :

        file_open_type = "w" if starting_model_index == 1 else "a"
        with open(data_path / "ModelTests" / "ModelTestSummary.txt", file_open_type) as f:
            for c in range(starting_model_index, model_count+1):
                error_sum_array = np.zeros(10)
                current_filename = f"ModelTest{c+1}-DX.txt"

                # with open(data_path / "ModelTests" / current_filename, "r") as f2:
                for i in range(10):
                    for j in range(len(test_usernames)):
                        error_value = extract_array_value_from_file(current_filename, index=i, user_num=j+1)
                        error_value = error_value if error_value and not np.isnan(error_value) else 0
                        error_sum_array[i] += error_value

                f.write(f"{current_filename}\n{error_sum_array}\n\n")

    def get_user_db(self, user, shows_to_take, with_mean=False):
        aff_db_entry_creator = AffDBEntryCreator(user, self.data, self.tags)
        aff_db_entry_creator.create_db_entries_from_user_list(shuffle_list=False, shows_to_take=shows_to_take,
                                                              db_type=1, for_predict=True)

        user_db_entries = aff_db_entry_creator.aff_db_entry_dict
        # user_mean_score = user_db_entries['Mean Score'][0]

        user_shows_df_with_name = pd.DataFrame(user_db_entries)
        # names = user_shows_df_with_name['Show Name'].to_list()
        # user_shows_df = user_shows_df_with_name.drop('Show Name', axis=1)
        # if 'User Score' in user_shows_df.columns: #remove this later, also the user score column from watched+ in affinitydb
        #     user_shows_df = user_shows_df.drop('User Score', axis=1)
        #
        # user_shows_df = user_shows_df.fillna(0)
        # normalized_df = self.aff_db.normalize_aff_df(user_shows_df, for_predict=True)
        # normalized_df = normalized_df.fillna(0)
        #
        # if not with_mean:
        #     normalized_df = normalized_df.drop('Show Score', axis=1)
        #     # cols_to_take = [x for x in normalized_df.columns if not x.startswith("Doubles-") #re.match(r'^Doubles-\d+$', x)
        #     #                 and x != 'Show Score' and x != 'Show Popularity'] #and 'Tag Count' not in x]
        #     # cols_to_take = [x for x in normalized_df.columns if x!= 'Show Score']
        #     # cols_to_take =
        #     # normalized_df = normalized_df[cols_to_take]
        #     normalized_df = AffinityDB.filter_df_for_model(normalized_df)

        return user_shows_df_with_name #,normalized_df

    # def fetch_predictions(self, normalized_df, user_shows_df_with_name, user, user_row, shows_to_take):
    def fetch_predictions(self, user_shows_df_with_name, user, user_row, shows_to_take):

        def format_predictions(preds, show_names):

            rounded_preds = [round(x, 1) for x in preds]
            pred_dict = {x[0]: x[1] for x in sorted(list(zip(show_names,rounded_preds)))}
            return pred_dict

        def calculate_prediction_adjustment(main_entry_predicted_score, main_entry, max_score_entry, show_predicted_score):
            """
            The logic behind the function is to lower the predictions of any show whose
            main entry was given a significantly lower rating compared to a later entry
            (which is ignored if we go by the general formula, which is simply to take
            the highest prediction among all entries).

            Example on what's possibly the worst offender - Gintama. Gintama's tags seem
            to VASTLY differ between the seasons, with the first season having tags like
            Gag Humor and Surreal Comedy, and the later ones having more action-oriented
            tags. So on my list, I get a prediction of 6.3 for the original Gintama (which makes
            sense according to my tastes) and a whopping 9.4 for later seasons.

            To solve the issue, we obviously need to take the main show's prediction into account.
            This is by no means a perfect solution, but going by the logic that the first season
            of a show does define it in most cases, there are more merits than demerits to it.

            So in short, we calculate two values :

            :param m1 - The difference between the MAL score of the main entry, and its predicted score.
            This tells us how compatible the model thinks the main show is with the user.
            :param m2 - The difference between the maximum predicted score of all the entries of a show,
            and the main entry's predicted score. A large difference would point to the existence of an entry
            that deviates strongly in terms of features (or quality) from the main entry. A small one means
            there's no issue - all the predictions are in the same range.
            :param m3 - The difference between the user score of the main entry, and its predicted score.
            Safeguard so that the m1-m2 system does not lower every prediction for people with very
            low mean scores (will elaborate at the bottom).

            From here,  we have two cases for m1 and m2:

            Case 1 - both m1 and m2 are high negative values. This is exactly the case we want to avoid -
            there are most likely one or two entries that are significantly different from the main show
            which are predicted to be well-liked by the user, but those entries aren't the main essence
            of the show - and since m1 is also high, the main essence of the show will not be liked by the user. So we want the prediction
            to be pushed down. In my Gintama example, m1 = 6.3 - 8.9 = -2.6, m2 = 6.3 - 9.4 = -3.1.

            Case 2 - m2 is high negative, but m1 isn't (or maybe it's even positive).
            There still is some discrepancy between individual entries of a show according
            to the model, but the main entry is well-received by the user - reducing the prediction
            just because a future entry is even MORE well-received is most likely undesirable in most
            cases.

            m3 and m4 are just there to avoid cases where a person with a 4.00 mean score gets every prediction
            lowered because their m1 will always be high, since pretty much every MAL score is much higher
            than their score (m3) and to avoid lowering already low predictions (lower than the user's mean score).

            This is by no means a perfect system, but it vastly improved the model's original predictions,
            and is much better than taking the average of all entries (due to outliers and things like OVAs).

            """

            main_entry_MAL_score = self.data.mean_score_per_show[main_entry]
            show_MAL_score = self.data.mean_score_per_show[max_score_entry]
            m1 = min(main_entry_predicted_score - main_entry_MAL_score, show_predicted_score - show_MAL_score)
            m2 = main_entry_predicted_score - show_predicted_score
            m3 = main_entry_predicted_score - user.mean_of_watched - 1
            m4 = user.mean_of_watched - show_predicted_score
            return max(min(0, m1), min(0,m2), min(0,m3), min(0, m4))  # We never want to increase the score hence min(0,m1)

        def process_predictions():
            max_pos_affs_per_entry = {
                user_shows_df_with_name['Show Name'][i]: (user_shows_df_with_name['Single Max Pos Affinity'][i]
                                                          + user_shows_df_with_name['Double Max Pos Affinity'][i])
                for i in range(len(user_shows_df_with_name))}
            predictions_per_show = {}
            adj_per_show = {}
            max_pos_affs_per_show = {}
            processed_entries = []
            for entry in user.entry_list:
                # change this to go in diff order
                if entry in processed_entries:
                    continue
                try:
                    main_show = self.tags.entry_tags_dict[entry]['Main']
                    main_score = predictions[main_show]
                    max_score = main_score
                    max_score_entry = entry
                    max_pos_aff = max_pos_affs_per_entry[main_show]
                except KeyError:
                    continue
                for entry, length_coeff in self.tags.show_tags_dict[main_show]['Related'].items():
                    if length_coeff == 1 and entry in user.entry_list and predictions[entry] > max_score:
                        max_score = predictions[entry]
                        max_score_entry = entry
                        max_pos_aff = max_pos_affs_per_entry[entry]
                    processed_entries.append(entry)
                predictions_per_show[main_show] = max_score + calculate_prediction_adjustment(main_score,
                                                  main_show, max_score_entry, max_score)
                adj_per_show[main_show] = calculate_prediction_adjustment(main_score,
                                                   main_show, max_score_entry, max_score)

                max_pos_affs_per_show[main_show] = round(max_pos_aff, 3)
            return predictions_per_show, max_pos_affs_per_show

        user_shows_df = user_shows_df_with_name.drop('Show Name', axis=1)
        normalized_df = self.aff_db.normalize_aff_df(user_shows_df, for_predict=True)
        normalized_df = normalized_df.fillna(0)

        # normalized_df = normalized_df.drop('Show Score', axis=1)
        normalized_df = AffinityDB.filter_df_for_model(normalized_df)

        if shows_to_take != "watched":
            features = normalized_df.values
        else:
            features = normalized_df.iloc[:, :-1].values

        predictions = self.model.predict(features)
        print("Right after predict")
        predictions = [min(10, float(x[0])) for x in predictions]
        # normalized_predictions = (predictions - np.mean(predictions)) / np.std(predictions)
        # normalized_df['Predictions'] = normalized_predictions
        # dev_features = normalized_df.values
        # dev_model = tf.keras.models.load_model(models_path / (self.model_filename.__str__().split("\\")[-1].split(".")[0] + "-Dev.h5"))
        # deviations = dev_model.predict(dev_features)
        # deviations = [x[0] for x in deviations]

        # normalized_dataset = AffinityDB.filter_df_for_model(deviation_training_dataset)
        # features = normalized_dataset.values
        # num_features = normalized_dataset.shape[1]
        # deviation_model = self.create_model_skeleton(num_features + 1)
        # predictions = self.model.predict(features)
        # predictions = [x[0] for x in predictions]
        # deviations = [(prediction - user_score) for prediction, user_score in list(zip(predictions, user_scores))]
        #
        # predictions = (predictions - np.mean(predictions)) / np.std(predictions)
        # normalized_dataset['Predictions'] = predictions
        # X_train, X_test, y_train, y_test = train_test_split(normalized_dataset, deviations, test_size=0.1,
        #                                                     random_state=42)
        # deviation_model.fit(
        #     x=np.float32(X_train),
        #     y=np.float32(y_train),
        #     epochs=self.epochs,
        #     validation_data=(np.float32(X_test), np.float32(y_test)),
        #     batch_size=512
        # )
        #
        # model_name = self.model_filename.__str__().split("\\")[-1].split(".")[0] + "-Dev.h5"
        # # The above line just gets the filename (T1-4-50-RSDD, for example) from the full path which
        # # is stored in self.model_filename. <--- Rename later
        # deviation_model.save(models_path / model_name)
        names = user_shows_df_with_name['Show Name'].to_list()
        # deviations2 = format_predictions(deviations, names)
        # deviations_sorted = {show: pred for show, pred in sorted(deviations2.items(), reverse=True, key=lambda x: x[1])}
        # adjusted_predictions = [pred - dev for pred, dev in list(zip(predictions, deviations))]
        # adjusted_predictions = format_predictions(adjusted_predictions,names)
        # adjusted_predictions_sorted = {show: pred for show, pred in
        #                                sorted(adjusted_predictions.items(), reverse=True, key=lambda x: x[1])}
        predictions = format_predictions(predictions, names)

        # model_overshoot = {show_name: 0 for show_name in predictions.keys()}
        #
        # for show_name, predicted_score in predictions.items():
        #     actual_score = user_row[show_name]
        #     model_overshoot[show_name] = predicted_score - actual_score


        # rounded_predictions = [round(x[0],1) for x in predictions]
        #
        # sorted_predictions_dict = {x[0]: x[1] for x in
        #                            sorted(list(zip(names, rounded_predictions)), key=lambda x: x[1])}

        predictions_per_show, max_pos_affs_per_show = process_predictions()

        # mean_scores_of_new_predictions = data.mean_score_per_show.select(predictions_per_show.keys()).to_numpy().tolist()[0]
        mean_scores_of_predictions = [self.tags.get_max_score_of_show(show, scores=self.data.mean_score_per_show)
                                          for show in predictions_per_show.keys()]

        user_watched_entries = user_row.select(user.entry_list)
        user_watched_entries = user_watched_entries.to_dict(as_series=False)
        user_watched_entries = {entry: score[0] for entry, score in user_watched_entries.items()}

        user_scores_per_show = [self.tags.get_max_score_of_show(show, scores=user_watched_entries)
                                          for show in predictions_per_show.keys()]

        # years_row = self.anime_db.df.row(self.anime_db.stats['Year'])
        # seasons_row = self.anime_db.df.row(self.anime_db.stats['Season'])

        years_dict = self.anime_db.years
        seasons_dict = self.anime_db.converted_seasons

        prediction_data = sorted(list(zip(predictions_per_show.items(), mean_scores_of_predictions,
                                          user_scores_per_show, max_pos_affs_per_show.values())),
                                 reverse=True, key=lambda x: (x[0][1], x[3]))

        new_predictions_list = []
        for pred in prediction_data:
            predictions_per_show = {}
            show_name = pred[0][0]
            predictions_per_show['ShowName'] = show_name
            predictions_per_show['PredictedScore'] = pred[0][1]
            predictions_per_show['UserScore'] = pred[2]
            predictions_per_show['MALScore'] = pred[1]
            predictions_per_show['Year'] = int(years_dict[show_name])
            predictions_per_show['Season'] = seasons_dict[show_name]

            show_single_tags = [d['name'] for d in self.tags.show_tags_dict[show_name]['Tags']]\
                                                  + self.tags.show_tags_dict[show_name]['Genres']
            predictions_per_show['Tags'] = show_single_tags
            new_predictions_list.append(predictions_per_show)

        new_predictions_list = sorted(new_predictions_list, key=lambda x: x['PredictedScore'], reverse=True)
        new_predictions_list_no_watched = [x for x in new_predictions_list if not x['UserScore']]
        print("Right after formatting")
        return new_predictions_list, new_predictions_list_no_watched

    @staticmethod
    def calculate_error(prediction_list, user_mean_score):

        errors = np.zeros(10)
        error_counts = np.zeros(10)

        pred_average = np.average([x['PredictedScore'] for x in prediction_list])
        pred_diff = user_mean_score - pred_average

        for pred_dict in prediction_list:
            if pred_dict['UserScore']:
                error_counts[pred_dict['UserScore'] - 1] += 1
                if pred_dict['UserScore'] > pred_dict['PredictedScore']:
                    errors[pred_dict['UserScore'] - 1] += abs(pred_dict['UserScore'] - pred_dict['PredictedScore'])

        for i in range(10):
            try:
                errors[i] /= error_counts[i]
                errors[i] -= pred_diff
            except ZeroDivisionError:
                errors[i] = 0
        return errors

    # def fetch_deviations(self):
    #     aff_db = pd.read_parquet(aff_db_path / "AffinityDB-P1-N.parquet")
    #     user_scores = list(aff_db['User Score'])
    #
    #     # user_shows_df = user_shows_df_with_name.drop('Show Name', axis=1)
    #     # normalized_df = self.aff_db.normalize_aff_df(user_shows_df, for_predict=True)
    #     # normalized_df = normalized_df.fillna(0)
    #     #
    #     # normalized_df = normalized_df.drop('Show Score', axis=1)
    #     normalized_df = AffinityDB.filter_df_for_model(aff_db)
    #     features = normalized_df.iloc[:, :-1].values
    #     predictions = self.model.predict(features)
    #     deviations = user_scores - predictions

    def predict_scores(self, user_name, db_type=1):

        with_mean = False
        shows_to_take = "all"
        # shows_to_take="all"
        user_row = self.user_db.get_user_db_entry(user_name)

        # turn user_row into a dict sometime when its not 3am
        # try:
        user = User(user_name, scores=user_row.select(self.data.relevant_shows).row(0, named=True),
                        scored_amount=user_row["Scored Shows"][0])
        # except AttributeError:
        #     raise UserListPrivateError("Error - User list was private or does not exist.", user_name)

        # user_shows_df_with_name, normalized_df = self.get_user_db(user, shows_to_take, with_mean)
        user_shows_df_with_name = self.get_user_db(user, shows_to_take)

        predictions, predictions_no_watched = self.fetch_predictions(user_shows_df_with_name,
                                                                     user, user_row, shows_to_take)

        print("Left fetch_predictions")
        predictions_sorted_by_diff = sorted(predictions, reverse=True, key=lambda x:(x['PredictedScore'] - x['MALScore']))

        predictions_no_watched_by_diff = sorted(predictions_no_watched, key=lambda x: (x['PredictedScore'] - x['MALScore']))
        # errors = self.calculate_error(predictions, user.mean_of_watched)
        # errors = [error if not np.isnan(error) else 0 for error in errors]
        print("Before returning from predict_scores")
        return predictions[0:400], predictions_sorted_by_diff[0:400]

    # def calculate_mean_pred_deviation(self):
    #     average_predicted_scores = {show_name: 0 for show_name in self.tags.show_tags_dict.keys()}
    #     average_actual_scores = {show_name: 0 for show_name in self.tags.show_tags_dict.keys()}
    #     prediction_counts = {show_name: 0 for show_name in self.tags.show_tags_dict.keys()}
    #     usernames = list(self.user_db.df['Username'])
    #     # for username in usernames:
    #     for i in range(0, len(usernames), 100):
    #         print(f"Currently on user {i}")
    #         try:
    #             errors, predictions = self.predict_scores(usernames[i])
    #         except UserListPrivateError as e:
    #             print(f"{e} Username: {e.username}")
    #             continue
    #         # if not average_predicted_scores:
    #         #     average_predicted_scores = {prediction}
    #         #     # average_predicted_scores = [0]*len(predictions)
    #         #     average_actual_scores = [0]*len(predictions)
    #         for j, prediction in enumerate(predictions):
    #             if prediction['UserScore']:
    #                 average_predicted_scores[prediction['ShowName']] += prediction['PredictedScore']
    #                 average_actual_scores[prediction['ShowName']] += prediction['UserScore']
    #                 prediction_counts[prediction['ShowName']] += 1
    #                 # add counter for amount of ratings for each show
    #
    #
    #     save_pickled_file(data_path / "test_avg_pred.pickle", average_predicted_scores)
    #     save_pickled_file(data_path / "test_avg_actual.pickle", average_actual_scores)

        # user_row = user_db.get_user_db_entry(user_name)
        # relevant_shows = list(tags.entry_tags_dict.keys())
        # user = User(user_name, scores=user_row.select(relevant_shows).row(0, named=True),
        #             scored_amount=user_row["Scored Shows"][0])

        # aff_db_entry_creator = AffDBEntryCreator(user, data, tags)
        # aff_db_entry_creator.create_db_entries_from_user_list(shuffle_list=False, shows_to_take=shows_to_take,
        #                                                       db_type=db_type, for_predict=True)
        #
        # user_db_entries = aff_db_entry_creator.aff_db_entry_dict
        # user_mean_score = user_db_entries['Mean Score'][0]
        #
        # user_shows_df_with_name = pd.DataFrame(user_db_entries)
        # names = user_shows_df_with_name['Show Name'].to_list()
        # user_shows_df = user_shows_df_with_name.drop('Show Name', axis=1)
        # user_shows_df = user_shows_df.fillna(0)
        # normalized_df = aff_db.normalize_aff_df(user_shows_df, for_predict=True)
        # normalized_df = normalized_df.fillna(0)
        #
        # if not with_mean:
        #     normalized_df = normalized_df.drop('Show Score', axis=1)
        #     cols_to_take = [x for x in normalized_df.columns if not x.startswith("Doubles-")
        #                     and x != 'Show Score' and x != 'Show Popularity']
        #     # cols_to_take = [x for x in normalized_df.columns if x!= 'Show Score']
        #     normalized_df = normalized_df[cols_to_take]

        # if shows_to_take != "watched":
        #     features = normalized_df.values
        # else:
        #     features = normalized_df.iloc[:, :-1].values

        # print(features)
        # predictions = model.predict(features)
        # predictions = [x[0] for x in predictions]
        # rounded_predictions = [round(x, 1) for x in predictions]
        # mean_scores_of_predicted_shows = data.mean_score_per_show.select(user.entry_list).to_numpy().tolist()[0]
        # mean_scores_of_predicted_shows = [tags.get_avg_score_of_show(show,data.mean_score_per_show)
        #                                   for show in user.entry_list]

        # pred_diffs_low = [round((prediction - mean_score),2) for (prediction, mean_score)
        #                   in list(zip(predictions, mean_scores_of_predicted_shows))]
        #
        # pred_diffs_balanced = [round(((prediction - mean_score) + (prediction - user.mean_of_watched))) for
        #                        (prediction, mean_score)
        #                        in list(zip(predictions, mean_scores_of_predicted_shows))]
        #
        # pred_diffs_high = [round(((prediction - mean_score) + (prediction - user.mean_of_watched)) * (mean_score - 5.5)) for
        #                    (prediction, mean_score)
        #                    in list(zip(predictions, mean_scores_of_predicted_shows))]



        # aff_sums = [get_aff_sum(show_name) for show_name in tags.show_tags_dict.keys()]

        # Advantage to lower scored shows
        # sorted_predictions_low = sorted(list(zip(names, pred_diffs_low, predictions, user_row[names].row(0)))
        #                                 , reverse=True, key=lambda x: x[1])
        #
        # # Balanced?...
        # sorted_predictions_balanced = sorted(list(zip(names, pred_diffs_balanced, predictions, user_row[names].row(0)))
        #                                      , reverse=True, key=lambda x: x[1])
        #
        # # Advantage to higher scored shows
        # sorted_predictions_high = sorted(list(zip(names, pred_diffs_high, predictions, user_row[names].row(0)))
        #                                  , reverse=True, key=lambda x: x[1])

        # sorted_predictions_high = sorted(list(zip(names, user_row[names].row(0), mean_scores_of_predicted_shows,
        #                                       rounded_predictions)), reverse=True, key=lambda x:(x[3], x[2]))
        #
        # sorted_predictions_low =  sorted(list(zip(names, user_row[names].row(0), mean_scores_of_predicted_shows,
        #                                       rounded_predictions)), reverse=True, key=lambda x:(x[3], -1*x[2]))
        #
        # sorted_predictions_balanced = sorted(list(zip(names, user_row[names].row(0), aff_sums,
        #                                       rounded_predictions)), reverse=True, key=lambda x:(x[3], x[2]))
        #
        # sorted_predictions_diff = sorted(list(zip(names, user_row[names].row(0), aff_sums,
        #                                       rounded_predictions, pred_diffs_low)), reverse=True, key=lambda x:(x[4], x[3],x[2]))

        # names = user_shows_df_with_name['Show Name'].to_list()
        # sorted_predictions_dict = {x[0]: x[1] for x in
        #                            sorted(list(zip(names, rounded_predictions)), key=lambda x: x[1])}

        # sorted_predictions_dict2 = {x[0]: x[1] for x in
        #                            sorted(list(zip(names, rounded_predictions)), key=lambda x: x[1], reverse=True)}

        # new_predictions_dict = {}
        # max_pos_affs_per_show = {}
        # max_pos_affs = {user_shows_df_with_name['Show Name'][i]: (user_shows_df_with_name['Single Max Pos Affinity'][i]
        #                                                        + user_shows_df_with_name['Double Max Pos Affinity'][i])
        #                 for i in range(len(user_shows_df_with_name))}
        # processed_entries = []
        # for entry in user.entry_list: #change this to go in diff order
        #     if entry in processed_entries:
        #         continue
        #     main_show = tags.entry_tags_dict[entry]['Main']
        #     try:
        #         max_score = sorted_predictions_dict[main_show]
        #         max_pos_aff = max_pos_affs[main_show]
        #     except KeyError:
        #         continue
        #     for entry, length_coeff in tags.show_tags_dict[main_show]['Related'].items():
        #         if length_coeff == 1 and entry in user.entry_list and sorted_predictions_dict[entry] > max_score:
        #             max_score = sorted_predictions_dict[entry]
        #             max_pos_aff = max_pos_affs[entry]
        #         processed_entries.append(entry)
        #     new_predictions_dict[main_show] = max_score
        #     max_pos_affs_per_show[main_show] = round(max_pos_aff,3)
        #
        # # mean_scores_of_new_predictions = data.mean_score_per_show.select(new_predictions_dict.keys()).to_numpy().tolist()[0]
        # mean_scores_of_new_predictions = [tags.get_max_score_of_show(show, data.mean_score_per_show)
        #                                   for show in new_predictions_dict.keys()]
        #
        # user_entry_list_row = user_row.select(user.entry_list)
        # user_scores_of_new_predictions = [tags.get_max_score_of_show(show, user_entry_list_row)
        #                                   for show in new_predictions_dict.keys()]
        #
        # new_predictions = sorted(list(zip(new_predictions_dict.items(), mean_scores_of_new_predictions,
        #                                   user_scores_of_new_predictions, max_pos_affs_per_show.values())), reverse=True, key=lambda x: (x[0][1], x[3]))

        # new_predictions_reverse1 = sorted(new_predictions, key = lambda x: (x[0][1], -1*x[1]), reverse=True)
        #
        # new_predictions_reverse2 = sorted(new_predictions, key=lambda x: (-1 * x[0][1], -1 * x[1]), reverse=True)

        # new_predictions_list = []
        # for pred in new_predictions:
        #     new_predictions_dict = {}
        #     new_predictions_dict['Show Name'] = pred[0][0]
        #     new_predictions_dict['Predicted Score'] = pred[0][1]
        #     new_predictions_dict['User Score'] = pred[2]
        #     new_predictions_dict['MAL Score'] = pred[1]
        #     new_predictions_list.append(new_predictions_dict)
        #
        # new_predictions_list = sorted(new_predictions_list, key=lambda x: x['Predicted Score'], reverse=True)
        # new_predictions_list_no_watched = [x for x in new_predictions_list if not x['User Score']]

        # new_predictions_dict_no_watched = {k: v for k, v in new_predictions_dict.items() if not v[1]}
        # new_predictions_dict = {x[0][0]: (x[0][1], x[2], x[3], x[1]) for x in new_predictions}

        # new_predictions_no_watched = {k: v for k, v in new_predictions_dict.items() if not v[1]}

        # print(calculate_error2())








