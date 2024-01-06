import tensorflow as tf
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
from UserDB import UserDB
from Tags import Tags
from AnimeDB import AnimeDB
from AffinityDB import AffDBEntryCreator, GeneralData, User, AffinityDB, UserAffinityCalculator
from Model import Model

#
# def get_single_tags():
#     tags = Tags()
#     counter_dict = {tag: 0 for tag in tags.all_anilist_tags}
#     for show in tags.show_tags_dict.keys():
#         tags_genres = tags.show_tags_dict[show]['Tags'] + tags.show_tags_dict[show]['Genres']
#         for tag in tags_genres:
#             try:
#                 tag_name = tag['name']
#                 if tag_name not in tags.all_anilist_tags:
#                     continue
#                 p = tags.adjust_tag_percentage(tag['percentage'])
#             except TypeError:
#                 tag_name = tag
#                 p = 1
#             counter_dict[tag_name] += p
#     counter_dict_sorted = {tag: count for tag, count in
#                            sorted(counter_dict.items(), reverse=True, key=lambda x: x[1])}
#     return counter_dict_sorted
#
#
# def get_double_tags():
#     tags = Tags()
#     counter_dict_doubles = {f"<{tag1}>x<{tags.all_anilist_tags[j]}>": 0 for i, tag1 in enumerate(tags.all_anilist_tags) for
#                             j in range(i + 1, len(tags.all_anilist_tags))}
#     for show in tags.show_tags_dict.keys():
#         tags_genres = tags.show_tags_dict[show]['Tags'] + tags.show_tags_dict[show]['Genres']
#         for i, tag1 in enumerate(tags_genres):
#             try:
#                 tag1_name = tag1['name']
#                 if tag1_name not in tags.all_anilist_tags:
#                     continue
#                 p1 = tags.adjust_tag_percentage(tag1['percentage'])
#             except TypeError:
#                 tag1_name = tag1
#                 p1 = 1
#
#             for j in range(i + 1, len(tags_genres)):
#                 tag2 = tags_genres[j]
#                 try:
#                     tag2_name = tag2['name']
#                     if tag2_name not in tags.all_anilist_tags:
#                         continue
#                     p2 = tags.adjust_tag_percentage(tag2['percentage'])
#                 except TypeError:
#                     tag2_name = tag2
#                     p2 = 1
#
#                 try:
#                     counter_dict_doubles[f"<{tag1_name}>x<{tag2_name}>"] += min(p1,p2)
#                 except KeyError:
#                     counter_dict_doubles[f"<{tag2_name}>x<{tag1_name}>"] += min(p1,p2)
#
#     counter_dict_sorted = {tag: count for tag, count in
#                            sorted(counter_dict_doubles.items(), reverse=True, key=lambda x: x[1])}
#     return counter_dict_sorted

if __name__ == '__main__':
    # tags = Tags()
    # test = pl.read_parquet("OG_tag_aff_db.parquet")
    # print(5)
    # load_affinity_DB2(shuffle=True)
    # aff_db = AffinityDB()
    # k = aff_db.df
    # aff_db = pl.read_parquet(f"{aff_db_filename}-N.parquet")
    # for col in aff_db.columns:
    #     # Calculate mean and standard deviation without zeros
    #     mean = aff_db.loc[aff_db[col] != 0, col].mean()
    #     std = aff_db.loc[aff_db[col] != 0, col].std()
    #
    #     # Normalize non-zero values
    #     aff_db.loc[aff_db[col] != 0, col] = (aff_db.loc[aff_db[col] != 0, col] - mean) / std
    # k = aff_db.df
    # df = aff_db.df.to_pandas()
    # df = df.fillna(0)



    # amount_of_dfs = count_major_parts()
    # for i in range(14, amount_of_dfs):
    #     print(f"Normalizing chunk {i+1}")
    #     df = pd.read_parquet(f"Partials\\{aff_db_filename}-P{i+1}-SN.parquet")
    #     if df.isna().any().any():
    #         print("Warning, NaNs detected!!!")
    #         has_nans_per_column = df.isna().any()
    #         for col in df.columns:
    #             if has_nans_per_column[col]:
    #                 df[col].fillna(0, inplace=True)
    #         df.to_parquet(f"Partials\\{aff_db_filename}-P{i+1}-SN.parquet")
    #         i=i-1
    # aff_db = AffinityDB()
    # aff_db.shuffle()
    # df = pd.read_parquet("AffinityDB-20M-S3.parquet")
    # aff_db = AffinityDB()
    # test = aff_db.normalize_aff_df(df=df, is_aff_db=True)
    # data = load_pickled_file("general_data.pickle")
    # data.db_type = 1
    # save_pickled_file("general_data.pickle", data)
    # aff_db = AffinityDB()
    # # aff_db.create_minor_parts((3,10))
    # aff_db.create()

    # tags = Tags()
    # # aff_db = AffinityDB()
    # # aff_db.combine()
    # # aff_db.shuffle()
    # # aff_db.normalize()
    # # terminate_program()
    # t = tags.entry_tags_dict
    # t2 = tags.show_tags_dict
    # t3 = tags.entry_tags_dict2
    # all_tags = tags.all_anilist_tags
    user_db = UserDB()
    names = user_db.df['Username']
    name_index = random.randint(0, 150000)
    # name_index = 36864
    name = names[name_index]
    model = Model("T4-1-50-RSDDP.h5")
    # model.train(epochs=50)
    # model.train(epochs=50)
    print(name)
    name = "BaronBrixius"
    model.predict_scores(name, db_type=1)


    # print(5)
    # tags = Tags()
    # my_model = tf.keras.models.load_model('my_model2.h5')
    # user_db = UserDB()

    # name = "BaronBrixius"
    # predict_scores(name, my_model,db_type=1)
    # terminate_program()

    # # normalize_shuffled_dfs()
    # aff_db = pq.ParquetFile(f"Partials\\{aff_db_filename}-P1-S.parquet")
    # num_features = len(aff_db.schema.names) - 1
    # print("Starting model creation")
    #
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(num_features, input_shape=(num_features,)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Activation('relu'),
    #     tf.keras.layers.Dense(128),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Activation('relu'),
    #     tf.keras.layers.Dense(128),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Activation('relu'),
    #     tf.keras.layers.Dense(1)  # Output layer for regression
    # ])
    #
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    #
    # model.compile(optimizer=optimizer,
    #               loss='mean_absolute_error',
    #               metrics=['mean_absolute_error'])
    #
    # print("Beginning training")
    # # train_on_chunk(model, num_training_examples, chunk_dataset)
    # train_on_chunk2(model, num_features)
    # model.save('T2-3.h5')



    # user_db = UserDB()
    # names = user_db.df['Username']
    # del user_db
    # for name in names:
    #     test = UserAffinityCalculator(
    # aff_db = AffinityDB()
    # aff_db = aff_db.df.to_pandas().sample(frac=1)
    print(5)
    # data = load_pickled_file("general_data.pickle")
    # df = pd.read_parquet(f"{aff_db_filename}.parquet")
    # df = normalize_df(df,is_aff_db=True,data=data)
    # user_name = "BaronBrixius"
    # tags = Tags()
    # # features = AffinityDBFeatures()
    # # data = load_pickled_file("general_data.pickle")
    # # aff_db_dict = AffinityDB.initialize_aff_db_dict(features, tags)
    # user_db = UserDB()
    # user_row = user_db.get_user_db_entry(user_name)
    # user = User(name=user_name, scores=user_row.select(aff_db.data.relevant_shows).row(0, named=True),
    #             scored_amount = user_row["Scored Shows"][0])
    # aff_db_entry_creator = AffDBEntryCreator(user, tags)
    # aff_db_entry_creator.create_db_entries_from_user_list()
    # user_db_entries = aff_db_entry_creator.aff_db_entry_dict
    # user_shows_df = pd.DataFrame(user_db_entries)
    # normalized_df = normalize_df(df=user_shows_df, data = aff_db.data, for_predict=True)
    # features = normalized_df.iloc[:, :-1].values
    # print(features)


    # # x = aff_db._load_affinity_DB(120)
    # # k = aff_db.df[1:3500000]
    # # anime_db = AnimeDB()
    # # k = user_db.scores_dict
    # # k = user_db.df
    # # K = user_db.filtered_df
    #

    # db_parts = os.cpu_count()/2
    # logger.debug("Starting new run")
    # create_df_in_parts(db_parts)
    # # load_affinity_DB(12)

