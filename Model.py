import pandas as pd
from filenames import *
import sys
from general_utils import *
from MAL_utils import *
from tensorflow import keras
import polars as pl
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import tensorflow as tf
from UserDB import UserDB
from Tags import Tags
from AnimeDB import AnimeDB
from AffinityDB import AffDBEntryCreator, GeneralData, User, AffinityDB, UserAffinityCalculator


class Model:

    def __init__(self, model_filename, batch_size=2048,user_name=None, with_mean=False, with_extra_doubles=False):
        self.model_filename = model_filename
        self.user_name = user_name
        self.batch_size = batch_size
        self.with_mean = with_mean
        self.with_extra_doubles = with_extra_doubles

    @staticmethod
    def _create_model(num_features):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1024, input_shape=(num_features,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.PReLU(),
            # tf.keras.layers.Dense(1024),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.PReLU(),
            # tf.keras.layers.Dense(1024),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(1)  # Output layer for regression
        ])
        return model

    def train(self, epochs):

        def interleave(files):
            return files.interleave(lambda file_path: tf.data.Dataset.from_generator(
                self._generate_batch, args=[file_path], output_signature=(
                    tf.TensorSpec(shape=(self.batch_size, num_features), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32))
            ), cycle_length=1)

        if os.path.exists(f"Partials\\{aff_db_filename}-P1-N.parquet"):
            aff_db = pq.ParquetFile(f"Partials\\{aff_db_filename}-P1-N.parquet")
        else:
            aff_db = AffinityDB()
            if not os.path.exists(f"Partials\\{aff_db_filename}-P1.parquet"):
                print("Creating Affinity DB")
                aff_db.create()
            print("Normalizing Affinity DB")
            aff_db.normalize()
            aff_db = pq.ParquetFile(f"Partials\\{aff_db_filename}-P1-N.parquet")

        num_features = len(aff_db.schema.names) - 1
        if not self.with_mean:
            num_features = 98
            if not os.path.exists(f"Partials\\{aff_db_filename}-P1-N-RSDD.parquet"):
                AffinityDB.remove_show_score()
        # if not self.with_extra_doubles:
        #     num_features -= 120
        #     if not os.path.exists(f"Partials\\{aff_db_filename}-P1-N-RSD.parquet"):
        #         AffinityDB.remove_extra_doubles()

        print("Starting model creation")

        model = self._create_model(num_features)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        model.compile(optimizer=optimizer,
                      loss='mean_absolute_error',
                      metrics=['mean_absolute_error'])

        file_amount = AffinityDB.count_major_parts()

        file_paths = [f"Partials\\{aff_db_filename}-P{i + 1}-N-RSDD.parquet" for i in range(file_amount)]
        files = tf.data.Dataset.list_files(file_paths)

        train_files = files.take(file_amount - 1)
        test_files = files.skip(file_amount - 1)

        train_dataset = interleave(train_files)
        test_dataset = interleave(test_files)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        print("Beginning training")

        model.fit(
            train_dataset,
            epochs=epochs,  # Number of times the entire dataset is passed forward and backward through the neural network
            validation_data=test_dataset,
            verbose=1,  # To view the training progress for each epoch
            callbacks=[callback]
        )
        # self.train_on_chunk(model, num_features)
        model.save(self.model_filename)

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

    def predict_scores(self, user_name, db_type=1, shows_to_ignore=None):
        def calculate_error(predictions):
            labels = normalized_df.iloc[:, -1].values.reshape(-1, 1)
            labels =  list(user_row[names].row(0))
            # predictions = [x[0] for x in predictions]
            rounded_predictions = [round(x) for x in predictions]
            rounded_diffs = [abs(x - y) for (x, y) in list(zip(labels, rounded_predictions)) if x]
            print(labels)
            # diffs = [abs(x - y) for (x, y) in list(zip(labels, predictions))]
            percentage_of_correct = len([x for x in rounded_diffs if not x]) / len(rounded_diffs) * 100
            mean_diffs = [abs(x - user_mean_score) for (x, y) in list(zip(labels, predictions)) if x]
            rounded_mean_diffs = [round(x) for x in mean_diffs]
            mean_percentage_of_correct = len([x for x in rounded_mean_diffs if not x]) / len(rounded_diffs) * 100
            return np.mean(rounded_diffs), np.mean(mean_diffs), np.mean(
                rounded_mean_diffs), percentage_of_correct, mean_percentage_of_correct

        def get_aff_sum(show_name):
            show_tags = tags.show_tags_dict[show_name]['Tags'] + tags.show_tags_dict[show_name]['DoubleTags']
            sum = 0
            tag_count = 0
            for tag in show_tags:
                if tag['name'] not in tags.all_anilist_tags:
                    continue
                p = tag['percentage']
                if p == 0:
                    break
                # aff = user.tag_affinity_dict[tag['name']]
                pos_aff = user.tag_pos_affinity_dict[tag['name']]
                sum += (pos_aff) * p
                tag_count += 1
            try:
                return sum / tag_count
            except ZeroDivisionError:
                return 0

        def calculate_error2():
            score_diffs = [x[0][1] - x[2] for x in new_predictions if x[2]]
            return np.mean(score_diffs)

        model = tf.keras.models.load_model(self.model_filename)
        with_mean = False
        tags = Tags()
        data = load_pickled_file("general_data.pickle")
        user_db = UserDB()
        aff_db = AffinityDB()
        shows_to_take = "all"

        user_row = user_db.get_user_db_entry(user_name)
        relevant_shows = list(tags.entry_tags_dict.keys())
        user = User(user_name, scores=user_row.select(relevant_shows).row(0, named=True),
                    scored_amount=user_row["Scored Shows"][0])

        aff_db_entry_creator = AffDBEntryCreator(user, data, tags)
        aff_db_entry_creator.create_db_entries_from_user_list(shuffle_list=False, shows_to_take=shows_to_take,
                                                              db_type=db_type, for_predict=True)

        user_db_entries = aff_db_entry_creator.aff_db_entry_dict
        user_mean_score = user_db_entries['Mean Score'][0]

        user_shows_df_with_name = pd.DataFrame(user_db_entries)
        names = user_shows_df_with_name['Show Name'].to_list()
        user_shows_df = user_shows_df_with_name.drop('Show Name', axis=1)
        user_shows_df = user_shows_df.fillna(0)
        normalized_df = aff_db.normalize_aff_df(user_shows_df, for_predict=True)
        normalized_df = normalized_df.fillna(0)

        if not with_mean:
            normalized_df = normalized_df.drop('Show Score', axis=1)
            cols_to_take = [x for x in normalized_df.columns if not x.startswith("Doubles-")
                            and x != 'Show Score']
            normalized_df = normalized_df[cols_to_take]

        if shows_to_take != "watched":
            features = normalized_df.values
        else:
            features = normalized_df.iloc[:, :-1].values

        # print(features)
        predictions = model.predict(features)
        predictions = [x[0] for x in predictions]
        mean_scores_of_predicted_shows = data.mean_score_per_show.select(user.entry_list).to_numpy().tolist()[0]
        # mean_scores_of_predicted_shows = [tags.get_avg_score_of_show(show,data.mean_score_per_show)
        #                                   for show in user.entry_list]

        pred_diffs_low = [round((prediction - mean_score),2) for (prediction, mean_score)
                          in list(zip(predictions, mean_scores_of_predicted_shows))]

        pred_diffs_balanced = [round(((prediction - mean_score) + (prediction - user.mean_of_watched))) for
                               (prediction, mean_score)
                               in list(zip(predictions, mean_scores_of_predicted_shows))]

        pred_diffs_high = [round(((prediction - mean_score) + (prediction - user.mean_of_watched)) * (mean_score - 5.5)) for
                           (prediction, mean_score)
                           in list(zip(predictions, mean_scores_of_predicted_shows))]

        rounded_predictions = [round(x,1) for x in predictions]

        aff_sums = [get_aff_sum(show_name) for show_name in tags.show_tags_dict.keys()]

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

        sorted_predictions_high = sorted(list(zip(names, user_row[names].row(0), mean_scores_of_predicted_shows,
                                              rounded_predictions)), reverse=True, key=lambda x:(x[3], x[2]))

        sorted_predictions_low =  sorted(list(zip(names, user_row[names].row(0), mean_scores_of_predicted_shows,
                                              rounded_predictions)), reverse=True, key=lambda x:(x[3], -1*x[2]))

        sorted_predictions_balanced = sorted(list(zip(names, user_row[names].row(0), aff_sums,
                                              rounded_predictions)), reverse=True, key=lambda x:(x[3], x[2]))

        sorted_predictions_diff = sorted(list(zip(names, user_row[names].row(0), aff_sums,
                                              rounded_predictions, pred_diffs_low)), reverse=True, key=lambda x:(x[4], x[3],x[2]))

        sorted_predictions_dict = {x[0]: x[1] for x in
                                   sorted(list(zip(names, rounded_predictions)), key=lambda x: x[1])}

        sorted_predictions_dict2 = {x[0]: x[1] for x in
                                   sorted(list(zip(names, rounded_predictions)), key=lambda x: x[1], reverse=True)}

        new_predictions_dict = {}
        max_pos_affs_per_show = {}
        max_pos_affs = {user_shows_df_with_name['Show Name'][i]: (user_shows_df_with_name['Single Max Pos Affinity'][i]
                                                               + user_shows_df_with_name['Double Max Pos Affinity'][i])
                        for i in range(len(user_shows_df_with_name))}
        processed_entries = []
        for entry in user.entry_list: #change this to go in diff order
            if entry in processed_entries:
                continue
            main_show = tags.entry_tags_dict[entry]['Main']
            try:
                max_score = sorted_predictions_dict[main_show]
                max_pos_aff = max_pos_affs[main_show]
            except KeyError:
                continue
            for entry, length_coeff in tags.show_tags_dict[main_show]['Related'].items():
                if length_coeff == 1 and entry in user.entry_list and sorted_predictions_dict[entry] > max_score:
                    max_score = sorted_predictions_dict[entry]
                    max_pos_aff = max_pos_affs[entry]
                processed_entries.append(entry)
            new_predictions_dict[main_show] = max_score
            max_pos_affs_per_show[main_show] = round(max_pos_aff,3)

        # mean_scores_of_new_predictions = data.mean_score_per_show.select(new_predictions_dict.keys()).to_numpy().tolist()[0]
        mean_scores_of_new_predictions = [tags.get_max_score_of_show(show, data.mean_score_per_show)
                                          for show in new_predictions_dict.keys()]

        user_entry_list_row = user_row.select(user.entry_list)
        user_scores_of_new_predictions = [tags.get_max_score_of_show(show, user_entry_list_row)
                                          for show in new_predictions_dict.keys()]

        new_predictions = sorted(list(zip(new_predictions_dict.items(), mean_scores_of_new_predictions,
                                          user_scores_of_new_predictions, max_pos_affs_per_show.values())), reverse=True, key=lambda x: (x[0][1], x[3]))

        new_predictions_reverse1 = sorted(new_predictions, key = lambda x: (x[0][1], -1*x[1]), reverse=True)

        new_predictions_reverse2 = sorted(new_predictions, key=lambda x: (-1 * x[0][1], -1 * x[1]), reverse=True)

        new_predictions_dict = {x[0][0]: (x[0][1], x[2], x[3], x[1]) for x in new_predictions}

        new_predictions_no_watched = {k: v for k, v in new_predictions_dict.items() if not v[1]}

        print(calculate_error2())

        print(5)






