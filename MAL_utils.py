from __future__ import print_function

from polars import ColumnNotFoundError

from general_utils import *
from bs4 import BeautifulSoup
import datetime
import time
import pandas as pd
from enum import Enum
import logging
import csv
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from random import shuffle
try:
    import thread
except ImportError:
    import _thread as thread
from operator import itemgetter
from colorama import Fore
from sortedcollections import OrderedSet
import random
from filenames import *


class Seasons(Enum):
    # Saved for future use since we store seasons as numbers rather than strings
    winter = 1
    spring = 2
    summer = 3
    fall = 4


class MediaTypes(Enum):
    tv = 1
    movie = 2
    ova = 3
    special = 4
    ona = 5





# class AllData:
#
#     _instance = None
#
#     def __new__(cls, *args, **kwargs):
#         """The class is a Singleton - we only need one instance of it since its purpose is
#         to house and create on demand all the data structures that are used in this project."""
#         if cls._instance is None:
#             cls._instance = super().__new__(cls, *args, **kwargs)
#         return cls._instance
#
#     def __init__(self):
#         # All properties are loaded on demand
#         self.user_df = UserDB()
#         self.anime_df = AnimeDB()
#         self.tags = Tags()
#         self.affinity_df = AffinityDB()


# class UserDB:
#     _instance = None
#
#     def __new__(cls, *args, **kwargs):
#         """The class is a Singleton - we only need one instance of it since its purpose is
#         to house and create on demand all the data structures that are used in this project."""
#         if cls._instance is None:
#             cls._instance = super().__new__(cls, *args, **kwargs)
#         return cls._instance
#
#     def __init__(self):
#         # All properties are loaded on demand
#         self._df=None
#         self._MAL_user_list = None
#         self._blacklist = None
#         self._scores_dict = None
#         self._columns=None
#         self._schema_dict=None
#         self.anime_db = AnimeDB()
#
#     stats = ["Index", "Username", "Mean Score", "Scored Shows"]
#
#     @property
#     def df(self):
#         """Polars database (stored as .parquet), contains username
#         + mean score + scored shows + all the scores of each user in it."""
#         if not isinstance(self._df, pl.DataFrame):
#             try:
#                 print("Loading main database")
#                 self._df = pl.read_parquet(main_database_name)
#             except FileNotFoundError:
#                 print("Main database not found. Creating new main database")
#                 amount = int(input("Insert the desired amount of users\n"))
#                 self._df = pl.DataFrame(schema=self.schema_dict)
#                 self.fill_main_database(amount)
#         return self._df
#
#     @df.setter
#     def df(self, value):
#         """Not really necessary, here just in case"""
#         if isinstance(value, pl.DataFrame):
#             self._df = value
#         else:
#             raise ValueError("df must be a Polars database")
#
#     @property
#     def scores_dict(self):
#         """A dictionary which holds the usernames of everyone in the database as keys,
#         and their respective score arrays as values. Saved in a pickle file.
#         Exists to significantly speed up real-time computation - working with the
#         Polars database would require casting and slicing during computation."""
#         if not self._scores_dict:
#             print("Unpickling main scores dictionary")
#             try:
#                 self._scores_dict = load_pickled_file(scores_dict_filename)
#             except FileNotFoundError:
#                 print("Pickled dictionary not found. Checking for main database")
#                 if not isinstance(self._df, pl.DataFrame) or self._df.is_empty():
#                     print("Main database not found, returning empty dictionary")
#                     self._scores_dict = {}
#                     # return self._scores_dict
#                 else:
#                     print("Confirmed existence of main database. Converting to dictionary")
#                     # This should never happen unless dict got manually deleted
#                     self.__main_db_to_pickled_dict()
#                     self._scores_dict = load_pickled_file(scores_dict_filename)
#             except EOFError:
#                 # This should never happen after the debugging stage is over
#                 print("Pickle file is corrupted, converting main database.")
#                 self.__main_db_to_pickled_dict()
#                 self._scores_dict = load_pickled_file(scores_dict_filename)
#         return self._scores_dict
#
#     @scores_dict.setter
#     def scores_dict(self, value):
#         """Not really necessary, here just in case"""
#         if isinstance(value, dict):
#             self._scores_dict = value
#         else:
#             raise ValueError("scores_dict must be a dictionary")
#
#     def save_scores_dict(self):
#         """Utility function just to have 1 line instead of 2"""
#         with open(scores_dict_filename, 'wb') as file:
#             pickle.dump(self.scores_dict, file)
#
#     def __main_db_to_pickled_dict(self):
#         """This function should never be used, it is only there for debug emergencies such as
#         the pickle file getting corrupted. Will take **A LOT** of memory and time because the
#         transpose function will turn our uint8 db into float64/utf8."""
#         score_df = self.main_df.select(self.titles)
#         usernames = self.main_df["Username"]
#         scores_dict = score_df.transpose(column_names=usernames).to_dict(as_series=False)
#         self._scores_dict = {key: list_to_uint8_array(value) for key, value in scores_dict.items()}
#         with open(scores_dict_filename, 'wb') as file:
#             pickle.dump(self.scores_dict, file)
#
#     @property
#     def MAL_users_list(self):
#         """CSV list of all the usernames in the database. Not strictly necessary anymore,
#         currently serves as backup."""
#         if not self._MAL_user_list:
#             print("Loading users list")
#             try:
#                 with open(MAL_users_filename, newline="", encoding='utf-8') as f:
#                     self._MAL_user_list = next(csv.reader(f, delimiter=','))
#                     print(f"{MAL_users_filename} loaded successfully")
#                     usernames = list(self.df['Username'])
#                     if len(self._MAL_user_list) != len(usernames):
#                         print("User list data was not up-to-date, returning updated user list")
#                         save_list_to_csv(usernames, MAL_users_filename)
#                         return usernames
#                     return self._MAL_user_list
#             except FileNotFoundError as e:
#                 print("Unable to load list - CSV file not found. Returning empty list")
#                 self._MAL_user_list = []
#         return self._MAL_user_list
#
#
#     @property
#     def blacklist(self):
#         """List of all users that didn't meet the requirements to have their lists
#         in the database. Stored here to avoid wasting an API request on checking
#         their lists again."""
#         if not self._blacklist:
#             print("Loading blacklist")
#             try:
#                 with open(blacklist_users_filename, newline="", encoding='utf-8') as f:
#                     self._blacklist = next(csv.reader(f, delimiter=','))
#                     print(f"{blacklist_users_filename} loaded successfully")
#                     return self._blacklist
#             except FileNotFoundError as e:
#                 print("Unable to load blacklist - file not found. Returning empty list")
#                 self._blacklist = []
#         return self._blacklist
#
#     @property  # change this into a normal var?
#     def schema_dict(self):
#         """The type schema of the main Polars database."""
#         if not self._schema_dict:
#             self._schema_dict = {'Index': pl.Int64, 'Username': pl.Utf8, 'Mean Score': pl.Float32,
#                                  'Scored Shows': pl.UInt32} | \
#                                 {x: y for (x, y) in zip(self.anime_db.titles, [pl.UInt8] * len(self.anime_db.titles))}
#         return self._schema_dict
#
#     @property
#     def columns(self):
#         """Columns of the main database"""
#         if not self._columns:
#             self._columns = ['Index', 'Username', 'Mean Score', 'Scored Shows'] + self.anime_db.titles
#         return self._columns
#
#     def save_df(self):
#         """Used during the creation of the main database. Saves all relevant files
#         (main database, user list, blacklist and scores dictionary) every N created
#         entries as defined in fill_main_database."""
#         self._df.write_parquet(main_database_name)
#         usernames = list(self._df['Username'])
#         print(f"Saving MAL user list. Length is {len(usernames)}")
#         save_list_to_csv(usernames,MAL_users_filename)
#         print(f"Saving blacklist. Length is {len(self._blacklist)}")
#         save_list_to_csv(self._blacklist,blacklist_users_filename)
#         print(f"Saving scores dictionary")
#         self.save_scores_dict()
#         print("Finished saving")
#
#     @timeit
#     def fill_main_database(self, amount):
#         """ This function adds users to the main database. If empty, it will create a new one.
#         The "amount" parameter is the final desired user count. """
#
#         @timeit
#         def add_user_list_to_db_list(user_index, username, user_list):
#             """Takes a single user's list and creates a row for them in the main database
#             if user meets criteria (>50 scored shows, >30 days account_age, non-troll mean score).
#             returns True if user was added, False otherwise."""
#
#             current_time = datetime.datetime.now(datetime.timezone.utc)
#             account_age_threshold = datetime.timedelta(days=30)
#             min_scored_shows = 50  # User needs to have scored at least 50 shows to be part of the DB
#             user_scored_shows = count_scored_shows(user_list)
#
#             if user_scored_shows >= min_scored_shows:
#                 show_amount = 0
#                 score_sum = 0
#                 account_age_verified = False
#                 for anime in user_list:
#                     title = anime['node']['title']
#                     score = anime['list_status']['score']
#
#                     if score == 0:
#                         actual_user_index = user_index + initial_users + saved_so_far
#                         print(f'Finished processing {show_amount} shows of user {username} ({actual_user_index})')
#                         break
#
#                     else:
#                         if not account_age_verified:
#                             update_timestamp = anime['list_status']['updated_at']
#                             time_since_update = current_time - datetime.datetime.fromisoformat(update_timestamp)
#                             if time_since_update > account_age_threshold:
#                                 account_age_verified = True
#                         # We test whether the account is at least one month old by seeing if at least one
#                         # anime update was done more than a month ago.
#                         show_amount += 1
#
#                         if title in self.anime_db.titles:
#                             scores_db_dict[title][user_index] = score
#                         score_sum += score
#
#                 mean_score = round(score_sum / show_amount, 4)
#
#                 if 2 <= mean_score <= 9.7:  # First we filter by scores, < 2 and > 9.7 means will just clutter the data
#                     if not account_age_verified:  # If we couldn't verify through the anime list, we check the
#                         # user's page directly. We only want to use this as a last resort since unlike the
#                         # former it takes another API call to do so.
#                         account_age = check_account_age_directly(username)
#                         if account_age < account_age_threshold:
#                             print(f"{username}'s account is {account_age} old, likely a troll")
#                             return False
#                         print(f"{username}'s account is {account_age} old, user verified. Adding to database")
#                     scores_db_dict['Index'][user_index] = current_users
#                     scores_db_dict['Username'][user_index] = username
#                     scores_db_dict['Scored Shows'][user_index] = show_amount
#                     scores_db_dict['Mean Score'][user_index] = mean_score
#                     new_user_list = list_to_uint8_array([scores_db_dict[key][user_index]
#                                                          for key in self.anime_db.titles])
#                     self.scores_dict[username] = new_user_list
#                     return True
#                 else:
#                     print(f"{username} has no meaningful scores, proceeding to next user")
#                     return False
#             print(f"{username} has less than 50 scored shows, proceeding to next user")
#             return False
#
#         def save_data():
#
#             print(f"Saving database. Currently on {current_users} entries")
#             logger.debug(f"Saving database. Currently on {current_users} entries")
#
#             # First we create a df from the temporary dictionary. Then we concatenate
#             # it with the existing database.
#             nonlocal saved_so_far
#             temp_df = pl.DataFrame(scores_db_dict, schema=self.schema_dict)
#             saved_so_far += temp_df.shape[0]
#
#             try:
#                 self._df = pl.concat(
#                     [
#                         self._df,
#                         temp_df,
#                     ],
#                     how="vertical",
#                 )
#             except ValueError:
#                 self._df = temp_df
#
#             # After we concatenated the main and temp databases, we need to save all the
#             # necessary files (polars db, user list + blacklist and scores dict)
#             # to avoid losing data in case the program stops for whatever reason.
#             self.save_df()
#
#         save_data_per = 10
#         saved_so_far = 0
#         min_scored_amount = 75000
#         max_scored_amount = 500000
#
#         # We want to take usernames from shows that are not TOO popular, but also not too niche.
#         # The reason for this is that the update table of a niche show will include many troll accounts
#         # that basically either put everything in their list, or are score-boosting alts with many shows.
#         # The update table of a very popular show like Attack on Titan on the other hand, will include many
#         # people that have just started watching anime, and thus only have a few shows in their list + score-boosting
#         # alts for those shows specifically. Since the program runs for weeks, we want to avoid wasting time
#         # on filtering those as much as possible.
#
#         @timeit
#         def initialize_temp_scores_dict():
#             """ We use a dictionary to avoid filling the db inplace, which takes MUCH more time. The data
#             is saved every save_data_per entries, after which the dictionary is reset to None values to avoid
#             it getting huge and taking up memory."""
#             remaining_amount = amount - current_users
#             for key in self.columns:
#                 scores_db_dict[key] = [None] * min(save_data_per, remaining_amount)
#
#         # ------------------------------ Main function starts here ----------------------------
#
#         scores_db_dict = {}
#         print("Test")
#         """This dictionary exists as temporary storage for user lists. Instead creating a new polars row
#         and adding it to the database as we get it, we add it to this dictionary. Then, each
#         save_data_per entries, we convert that dictionary to a Polars dataframe and concatenate
#         it with the main one. This results in significant speedup."""
#
#         current_users = len(self.MAL_users_list)
#
#         print("Test 1.5")
#         initial_users = current_users
#
#         rows = self.anime_db.df.rows()
#         ids = rows[0][1:]
#         scored_amount = rows[2][1:]
#
#         print("Test 2")
#         initialize_temp_scores_dict()
#
#         print(f"Starting MAL user length/database size is : {len(self.MAL_users_list)}")
#         print(f"Starting Blacklist length is : {len(self.blacklist)}")
#
#         ids_titles_scored = list(zip(ids, self.anime_db.titles, scored_amount))
#         shuffle(ids_titles_scored)  # Shuffle to start getting users from a random show, not the top shows
#         # We start by iterating over the shows we have in our show DB. Then, for each show, we get 375 usernames.
#         while True:
#             for id, title, scored_amount in ids_titles_scored:
#                 if not scored_amount or int(scored_amount) < min_scored_amount \
#                         or int(scored_amount) > max_scored_amount:
#                     print(f"Scored amount of {title} is {scored_amount}, moving to next show")
#                     continue
#                 print(f"Scored amount of {title} is {scored_amount}, proceeding with current show")
#
#                 title = replace_characters_for_url(title)
#                 base_url = f"https://myanimelist.net/anime/{id}/{title}/stats?"
#                 print(base_url)
#                 users_table = get_usernames_from_show(base_url)
#                 # This returns a table of list updates
#
#                 for table_row in users_table:
#                     # The list update table includes usernames, timestamps and more.
#                     # We extract the usernames from there by their assigned CSS class.
#                     if current_users == amount:
#                         break
#
#                     user_link = table_row.findNext(
#                         "div", {"class": "di-tc va-m al pl4"}).findNext("a")
#                     user_name = str(user_link.string)  # This thing is actually a bs4.SoupString,
#                     # not a regular Python string.
#
#                     if user_name not in self.MAL_users_list and user_name not in self.blacklist:
#
#                         if user_name.startswith('ishinashi'):
#                             logger.debug("Retard troll detected, continuing on")
#                             continue
#
#                         user_anime_list = get_user_MAL_list(user_name, full_list=False)
#                         dict_user_index = current_users - initial_users - saved_so_far
#                         added = add_user_list_to_db_list(dict_user_index,
#                                                          user_name, user_anime_list)
#
#                         if added:
#                             current_users += 1
#                             self.MAL_users_list.append(user_name)
#                             if (current_users - initial_users) % save_data_per == 0:
#                                 # We save the database and the lists every n users just to be safe
#                                 save_data()
#                                 if current_users == amount:
#                                     return
#                                 initialize_temp_scores_dict()
#                         else:
#                             self.blacklist.append(user_name)
#                     else:
#                         print(f"{user_name} has already been added/blacklisted, moving on to next user")
#
#                 if current_users == amount:
#                     save_data()
#                     # We reached the necessary amount of users in the database
#                     return
#
#     def create_user_entry(self,user_name):
#         """Creates a user entry for the one we're calculating affinity for + returns a Polars row of their stats.
#         Currently not in use due to being time-inefficient at adding in real-time. New process maybe?"""
#         if user_name not in self.MAL_users_list and user_name not in self.blacklist:
#             user_list = get_user_MAL_list(user_name, full_list=False)
#             if not user_list:
#                 print("Unable to access private user list. Please make your list public and try again.")
#                 terminate_program()
#         else:
#             print("User already exists in database")
#             return
#
#         old_titles = [x for x in self.df.columns if x not in self.stats]
#
#         anime_indexes = {v: k for (k, v) in enumerate(old_titles)}
#         # new_list = [None] * len(self.titles)
#         new_list = [None] * (len(old_titles))
#
#         show_amount = 0
#         score_sum = 0
#         for anime in user_list:  # anime_list, scores_db
#             title = anime['node']['title']
#             score = anime['list_status']['score']
#             if score == 0:
#                 break
#             show_amount += 1
#             if title in self.anime_db.titles:
#                 new_list[anime_indexes[title]] = score
#             score_sum += score
#
#         mean_score = round(score_sum / show_amount, 4)
#         new_list_for_return = list_to_uint8_array(new_list)  # We save the original list here,
#         # because later we'll have to add fields that are relevant for adding to DB but not for list comparison.
#
#         if 2 <= mean_score <= 9.7:  # Removing troll accounts/people who score everything 10/10
#             index = self.df.shape[0]
#             new_list = [index,user_name,mean_score,show_amount] + new_list
#
#             schema_dict = {'Index' : pl.Int64, 'Username': pl.Utf8, 'Mean Score': pl.Float32, 'Scored Shows': pl.UInt32} | \
#                           {x: y for (x, y) in zip(old_titles, [pl.UInt8] * len(old_titles))}
#             # cols = ['Index', 'Username', 'Mean Score', 'Scored Shows'] + self.titles
#
#             new_dict = {k : v for k,v in zip(self.df.columns, new_list)}
#
#             new_row = pl.DataFrame(new_dict,schema=schema_dict)
#             print(new_row)
#
#             # new_row = synchronize_dfs(new_row,self.main_df)
#             self.df = pl.concat(
#                 [
#                     self.df,
#                     new_row,
#                 ],
#                 how="vertical",
#             )
#
#             self.df.write_parquet(main_database_name)
#             self.MAL_users_list.append(user_name)
#             save_list_to_csv(self.MAL_users_list,MAL_users_filename)
#         return new_list_for_return
#
#
# class AnimeDB:
#
#     _instance = None
#
#     def __new__(cls, *args, **kwargs):
#         """The class is a Singleton - we only need one instance of it since its purpose is
#         to house and create on demand all the data structures that are used in this project."""
#         if cls._instance is None:
#             cls._instance = super().__new__(cls, *args, **kwargs)
#         return cls._instance
#
#     def __init__(self):
#         # All properties are loaded on demand
#         self._df=None
#         self._titles = None
#         # self._user_df = None # Check this later
#
#     stats = {'ID': 0, 'Mean Score': 1, 'Scores': 2, 'Members': 3, 'Episodes': 4,
#                       'Duration': 5, 'Type': 6, 'Year': 7, 'Season': 8}
#
#     @property
#     def df(self):
#         """Database with info on all the anime that's listed (and ranked) on MAL.
#         Fields are as shown in the "required_fields" variable of generate_anime_DB."""
#         if not isinstance(self._df, pl.DataFrame):
#             try:
#                 print("Loading anime database")
#                 self._df = pl.read_parquet(anime_database_name)
#                 print("Anime database loaded successfully")
#             except FileNotFoundError:
#                 print("Anime database not found. Creating new anime database")
#                 self.generate_anime_DB()
#                 self._df = pl.read_parquet(anime_database_name)
#         return self._df
#
#     @property
#     def titles(self):  # change this into a normal var?
#         """A list of all the anime titles."""
#         if not self._titles:
#             self._titles = self.df.columns[1:]  # anime_df will automatically be generated
#             # for title in self._titles:
#             #     if title in self.unlisted_titles:
#             #         self._titles.remove(title)
#             # through the property if it doesn't exist yet
#         return self._titles
#
#     # def main_df(self):
#     #     if not isinstance(self._main_df, pl.Dataframe):
#     #         try:
#     #             print("Loading user database")
#     #             self._main_df = pl.read_parquet(main_database_name)
#     #         except FileNotFoundError:
#     #             pass # Inside the AnimeDB class, if the user DB doesn't exist, we don't have to load it.
#
#     def get_stats_of_shows(self,show_list, relevant_stats):
#         """ Will create a dictionary that has every show in show_list as the key, and every stat in relevant_stats
#             in a list as the value.
#             Example :
#             {'Shingeki no Kyojin': {'ID': 16498.0, 'Mean Score': 8.53, 'Members': 3717089.0},
#              'Shingeki no Kyojin OVA': {'ID': 18397.0, 'Mean Score': 7.87, 'Members': 439454.0}"""
#
#         stats_dict = {}
#         for show in show_list:
#             show_dict = {}
#             for stat in relevant_stats:
#                 try:
#                     show_dict[stat] = self.df.filter(pl.col('Rows') == stat)[show].item()
#                 except ColumnNotFoundError:
#                     break
#                 except ValueError:
#                     show_dict[stat] = None
#             if show_dict:
#                 stats_dict[show] = show_dict
#         return stats_dict
#
#     def generate_anime_DB(self,non_sequels_only=False):
#         """Creates the anime database which contains data (as defined in required_fields)
#         on every show"""
#         """""┌─────┬────────────┬────────────┬────────────┬───┬──────────┬────────────┬────────────┬────────────┐
#         │ Row ┆ Fullmetal  ┆ Shingeki   ┆ "Oshi no   ┆ … ┆ Kokuhaku ┆ Hametsu no ┆ Utsu       ┆ Tenkuu     │
#         │ s   ┆ Alchemist: ┆ no Kyojin: ┆ Ko"        ┆   ┆ ---      ┆ Mars       ┆ Musume     ┆ Danzai Ske │
#         │ --- ┆ Brotherhoo ┆ The Final  ┆ ---        ┆   ┆ f64      ┆ ---        ┆ Sayuri     ┆ lter+Heave │
#         │ str ┆ d          ┆ Se…        ┆ f64        ┆   ┆          ┆ f64        ┆ ---        ┆ n          │
#         │     ┆ ---        ┆ ---        ┆            ┆   ┆          ┆            ┆ f64        ┆ ---        │
#         │     ┆ f64        ┆ f64        ┆            ┆   ┆          ┆            ┆            ┆ f64        │
#         ╞═════╪════════════╪════════════╪════════════╪═══╪══════════╪════════════╪════════════╪════════════╡
#         │ ID  ┆ 5114.0     ┆ 51535.0    ┆ 52034.0    ┆ … ┆ 31634.0  ┆ 413.0      ┆ 13405.0    ┆ 3287.0     │
#         │ Mea ┆ 9.1        ┆ 9.08       ┆ 9.08       ┆ … ┆ 2.3      ┆ 2.22       ┆ 1.98       ┆ 1.84       │
#         │ n   ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ Sco ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ re  ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ Sco ┆ 2.004693e6 ┆ 140026.0   ┆ 134283.0   ┆ … ┆ 4847.0   ┆ 47441.0    ┆ 15790.0    ┆ 26203.0    │
#         │ res ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ Mem ┆ 3.151508e6 ┆ 409744.0   ┆ 404237.0   ┆ … ┆ 6602.0   ┆ 65349.0    ┆ 20649.0    ┆ 37022.0    │
#         │ ber ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ s   ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ Fav ┆ 216247.0   ┆ 8386.0     ┆ 12470.0    ┆ … ┆ 17.0     ┆ 297.0      ┆ 49.0       ┆ 115.0      │
#         │ ori ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ tes ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ Epi ┆ 64.0       ┆ 2.0        ┆ 11.0       ┆ … ┆ 1.0      ┆ 1.0        ┆ 1.0        ┆ 1.0        │
#         │ sod ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ es  ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ Dur ┆ 24.0       ┆ 61.0       ┆ 20.0       ┆ … ┆ 1.0      ┆ 19.0       ┆ 3.0        ┆ 19.0       │
#         │ ati ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ on  ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ Typ ┆ 1.0        ┆ 4.0        ┆ 1.0        ┆ … ┆ 5.0      ┆ 3.0        ┆ 3.0        ┆ 3.0        │
#         │ e   ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ Yea ┆ 2009.0     ┆ 2023.0     ┆ 2023.0     ┆ … ┆ 2015.0   ┆ 2005.0     ┆ 2003.0     ┆ 2004.0     │
#         │ r   ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         │ Sea ┆ 2.0        ┆ 1.0        ┆ 2.0        ┆ … ┆ 3.0      ┆ 3.0        ┆ 1.0        ┆ 4.0        │
#         │ son ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
#         └─────┴────────────┴────────────┴────────────┴───┴──────────┴────────────┴────────────┴────────────┘
#
#         """
#         def create_anime_DB_entry(anime, required_fields):
#             # Helper function, creates a list which will later serve as a column in the anime DB.
#
#             # def get_anime_duration(anime):
#             #     print(f"Currently on anime {anime['title']}")
#             #     text_duration = anime["duration"]
#             #     if not text_duration:
#             #         return None
#             #     if 'sec' in text_duration:
#             #         return 1
#             #     text_duration_split = text_duration.split()
#             #     if 'hr' in text_duration: # text_duration_split will be something like "1", "hr", "10", "min"
#             #         try:
#             #             return int(text_duration_split[0])*60 + int(text_duration_split[2])
#             #         except IndexError:  # This means the duration is "2 hr" or "1 hr" rather than "1 hr 15 min".
#             #             return int(text_duration_split[0])*60
#             #     else: #
#             #         return int(text_duration_split[0])
#
#             def parse_start_date(date_str):
#                 try:
#                     date_parsed = datetime.datetime.strptime(date_str, "%Y-%m-%d")
#                     return date_parsed.year, (date_parsed.month - 1) // 3 + 1
#                 except ValueError:
#                     try:
#                         date_parsed = datetime.datetime.strptime(date_str, "%Y-%m")
#                         return date_parsed.year, (date_parsed.month - 1) // 3 + 1
#                     except ValueError:
#                         return int(date_str), 1 # Only the year is known, season is assumed to be Winter (placeholder)
#
#
#             # anime_data = list((itemgetter(*required_fields))(anime))
#             anime_data = []
#             for field in fields_no_edit_needed: # This takes care of the fields that can be put into db as they are
#                 anime_data.append(anime["node"][field])
#
#             # The rest of the fields need editing or error handling
#             if anime["node"]["num_episodes"]:
#                 anime_data.append(anime["node"]["num_episodes"])
#             else: # This means episode_num = null, can only happen with VERY obscure shows.
#                 anime_data.append(1)
#
#             if anime["node"]["average_episode_duration"]:
#                 anime_data.append(round(anime["node"]["average_episode_duration"]/60,1))
#             else: # Same as above
#                 anime_data.append(20)
#
#             # try:
#             #     anime_data.append(get_anime_duration(anime)) # Duration needs separate handling
#             # except ValueError:
#             #     anime_data.append(20)  # Duration not listed (most likely very obscure show), setting default.
#
#             try:
#                 media_type_index = MediaTypes[anime['node']["media_type"]].value
#             except KeyError:
#                 media_type_index = None
#             anime_data.append(media_type_index)
#
#             try:
#                 year = int(anime["node"]["start_season"]["year"])
#                 season = anime["node"]["start_season"]["season"]
#                 season = Seasons[season].value # We convert the season to a numerical value for the database
#             except KeyError:
#                 try:
#                     year,season = parse_start_date(anime["node"]["start_date"])
#                 except KeyError:
#                     year, season = None, None
#             anime_data.append(year)
#             anime_data.append(season)
#
#             # print(anime["type"])
#             # anime_data.append(anime["aired"]["prop"]["from"]['year'])
#             # month = anime["aired"]["prop"]["from"]['month']
#             # try:
#             #     # season = Seasons((month - 1) // 3).name
#             #     season = (month - 1) // 3 + 1
#             # except TypeError:
#             #     season = None  # Sometimes the season isn't listed
#             # anime_data.append(season)
#
#             # title = anime_data[0]
#             # anime_data_dict[title] = anime_data[1:]
#
#             title = anime["node"]["title"]
#             anime_data_dict[title] = anime_data
#             return title  # Title is returned to check whether we reached the last show
#
#         # def synchronize_main_dbs():
#         #     # missing_cols = [x for x in self.titles if x not in self.main_df.columns]
#         #     # for col_name in missing_cols:
#         #     #     self.main_df = self.main_df.with_columns(
#         #     #         pl.Series(col_name, [None] * len(self.main_df), dtype=pl.UInt8))
#         #
#         #     missing_cols = [x for x in self.titles if x not in self.main_df.columns]
#         #     for col_name in missing_cols:
#         #         self.main_df = self.main_df.with_columns(
#         #             pl.Series(col_name, [None] * len(self.main_df), dtype=pl.UInt8))
#         #     self.main_df.write_parquet(main_database_name)
#
#         # def synchronize_dfs(df1,df2,cols1=None,cols2=None):
#         #     # Add all columns in db2 but not in db1 to db1
#         #     if cols1 and cols2:
#         #         missing_cols = [x for x in cols2 if x not in cols1]
#         #     else:
#         #         missing_cols = [x for x in df2.columns if x not in df1.columns]
#         #
#         #     for col_name in missing_cols:
#         #         df1 = df1.with_columns(
#         #             pl.Series(col_name, [None] * len(df1), dtype=pl.UInt8))
#         #
#         #     return df1
#             # self.main_df.write_parquet(main_database_name)
#         # synchronize_dfs()
#             # Also remove all titles that are in main but not in anime
#
#         last_show_reached = False
#         last_show = 'Tenkuu Danzai'
#         # Lowest rated show, it'll be the last one in the sorted list.
#         # Hardcoded because I'm 100% sure nothing will ever be rated
#         # lower than THAT in my lifetime. The order is a bit off using Jikan
#         # so checking for N/A score and stopping is not an option.
#
#         required_fields = ["title", "mal_id", "score",
#                            "scored_by", "members", "favorites"]
#
#         url_required_fields = ["id","mean","num_scoring_users","num_list_users","num_episodes", "average_episode_duration",
#                            "media_type","start_season", "start_date"]
#
#         fields_no_edit_needed = ["id", "mean", "num_scoring_users", "num_list_users"]
#
#         # The fields we need from the JSON object containing information about a single anime.
#
#         # anime_data_dict = {'Rows': ['ID', 'Mean Score', 'Scores',
#         #                             'Members', 'Favorites', 'Episodes', 'Duration', 'Type',  'Year', 'Season']}
#         #
#         anime_data_dict = {'Rows': ['ID', 'Mean Score', 'Scores',
#                                     'Members', 'Episodes', 'Duration', 'Type',  'Year', 'Season']}
#
#         page_num = 0
#         while not last_show_reached:
#             # We loop over pages of anime info that we get from the Jikan API (the info is
#             # sorted by score, from highest to lowest) until we reach the lowest rated show.
#             # There should be 25 shows per page/batch.
#
#             anime_batch = get_anime_batch_from_MAL(page_num, url_required_fields)
#             # url = f'https://api.jikan.moe/v4/top/anime?page={page_num}'
#             try:
#                 print(f"Currently on score {anime_batch['data'][-1]['node']['mean']}")
#             except KeyError:
#                 print("Finished")
#             while anime_batch is None:
#                 # If we failed to get the batch for some reason, we retry until it's a success.
#                 # Jikan API does not require authorization, so the only reason for a failure
#                 # could be an outage in the API itself, in which case we wouldn't want to
#                 # timeout/stop the function as an outage could technically last for a few hours,
#                 # or even days.
#                 print("Error - unable to get batch. Sleeping just to be safe, "
#                       "then trying again.")
#                 logging.error("Error - unable to get batch. Sleeping just to be safe, "
#                               "then trying again.")
#                 time.sleep(Sleep.LONG_SLEEP)
#                 # anime_batch = call_function_through_process(get_search_results, url)
#                 anime_batch = get_anime_batch_from_MAL(page_num, url_required_fields)
#                 print(anime_batch)
#
#             for anime in anime_batch["data"]:
#                 if not non_sequels_only or anime_is_valid(anime):
#                     title = create_anime_DB_entry(anime, url_required_fields)
#                 if title.startswith(last_show):
#                     last_show_reached = True
#                     break
#             page_num += 1
#         table = pa.Table.from_pydict(anime_data_dict)
#         pq.write_table(table, anime_database_name) # This creates a .parquet file from the dict
#         # if self._main_df:
#         #     synchronize_main_dbs()
#
#
# class Tags:
#
#     _instance = None
#
#     query = '''
#     query ($page: Int, $isadult : Boolean) {
#       Page(page: $page, perPage: 50) {
#         pageInfo {
#           total
#           perPage
#           currentPage
#           lastPage
#           hasNextPage
#         }
#         media (type : ANIME, isAdult : $isadult) {
#           title{
#             romaji
#           }
#           idMal
#           tags {
#             name
#             category
#             rank
#           }
#           genres
#           studios(isMain : true) {
#             nodes {
#                name
#             }
#           }
#           recommendations(page : 1, perPage : 25){
#             nodes{
#               rating
#               mediaRecommendation{
#                 idMal
#                 title{
#                   romaji
#                 }
#               }
#             }
#           }
#         }
#       }
#     }
#     '''
#
#     def __new__(cls, *args, **kwargs):
#         """The class is a Singleton - we only need one instance of it since its purpose is
#         to house and create on demand all the data structures that are used in this project."""
#         if cls._instance is None:
#             cls._instance = super().__new__(cls, *args, **kwargs)
#         return cls._instance
#
#     def __init__(self):
#         # All properties are loaded on demand
#         self._shows_tags_dict = {}
#         self._all_tags_list = OrderedSet()
#
#     @property
#     def all_tags_list(self):
#         if not self._all_tags_list:
#             self.get_full_tags_list()
#         return self._all_tags_list
#
#     @property
#     def shows_tags_dict(self):
#         if not self._shows_tags_dict:
#             try:
#                 print("Loading shows-tags dictionary")
#                 self._shows_tags_dict = load_pickled_file(shows_tags_filename)
#             except FileNotFoundError:
#                 print("Shows-tags dictionary not found. Creating new shows-tags dictionary")
#                 self.get_shows_tags()
#         return self._shows_tags_dict
#
#     def get_full_tags_list(self):
#         # Gets list of all tags + counts amount of show per studio
#         studio_dict = {}
#
#         with open("NSFWTags.txt", 'r') as file:
#             banned_tags = file.read().splitlines()
#
#         for show, show_dict in self.shows_tags_dict.items():
#             if 'Tags' in show_dict:
#                 for tag_dict in show_dict['Tags']:
#                     if tag_dict['name'] not in banned_tags:
#                         self._all_tags_list.add(tag_dict['name'])
#             if 'Genres' in show_dict.keys():
#                 for genre in show_dict['Genres']:
#                     self._all_tags_list.add(genre)
#             if 'Studio' in show_dict.keys():
#                 if show_dict['Studio']:
#                     if show_dict['Studio'] not in studio_dict:
#                         studio_dict[show_dict['Studio']] = 1
#                     else:
#                         studio_dict[show_dict['Studio']] += 1  # # #
#
#         extra_studios = ['Trigger', 'Passione', 'project No.9', 'POLYGON PICTURES', 'EMT Squared', 'SANZIGEN']
#
#         # Only keeps studios that have over 30 shows or are in extra_studios
#         for studio, amount_of_shows in studio_dict.items():
#             if amount_of_shows >= 30 or studio in extra_studios:
#                 self._all_tags_list.add(studio)
#
#         self._all_tags_list = list(self._all_tags_list)
#
#     def get_shows_tags(self):
#         def get_recommended_shows():
#             try:
#                 sorted_recs = sorted(media['recommendations']['nodes'], key=lambda x: x['rating'], reverse=True)
#             except KeyError:
#                 return None
#
#             rec_list_length = min(5, len(sorted_recs))
#             rec_dict = {}
#             for rec in sorted_recs[0:rec_list_length]:
#                 try:
#                     rec_index = ids.index(rec['mediaRecommendation']['idMal'])
#                 except (TypeError, ValueError):
#                     continue
#                 try:
#                     rec_MAL_title = anime_db.titles[rec_index]
#                 except IndexError:
#                     print(rec_index,media)
#                 # Can't take title directly from the recommendation object,
#                 # might be different from the MAL title we have in the database
#                 # rec_dict['Ratings'].append(rec['rating'])
#                 # rec_dict['Titles'].append(rec_MAL_title)
#                 rec_dict[rec_MAL_title] = rec['rating']
#             return rec_dict
#
#         anime_db=AnimeDB()
#         url = "https://graphql.anilist.co"
#         ids = anime_db.df.row(anime_db.stats['ID'])[1:]
#         has_next_page = True
#         page = 1
#         while has_next_page:
#             print(f"Currently on page {page}")
#             variables = {"page": page, "isAdult": False}
#             response = requests.post(url, json={"query": self.query, "variables": variables})
#             response = response.json()
#             media_list = response["data"]["Page"]["media"]
#             # categories = set()
#             for media in media_list:
#                 try:
#                     index = ids.index(media["idMal"])
#                 except (TypeError, ValueError):
#                     continue
#                 title = anime_db.titles[index]  # Check this, might not be synchronized!
#                 show_stats = anime_db.get_stats_of_shows([title], ["Episodes", "Duration"])
#                 show_recommendations = get_recommended_shows()
#
#                 if show_stats[title]["Episodes"] * show_stats[title]["Duration"] >= 15\
#                         and show_stats[title]["Duration"]>=2:
#                     result = {
#                         "Tags": [{"name": tag["name"], "percentage": tag["rank"], "category": tag["category"]} for tag
#                                  in
#                                  media["tags"]],
#                         "Genres": media["genres"],
#                         "Studio": media["studios"]["nodes"][0]["name"] if media["studios"]["nodes"] else None,
#                         "Recommended": show_recommendations
#                         # If recs amount less than 25 start penalizing?
#
#                     }
#
#                     # Separate studios from tags, genres too maybe?
#                     self._shows_tags_dict[title] = result
#             has_next_page = response["data"]["Page"]["pageInfo"]["hasNextPage"]
#             page = page + 1
#             time.sleep(1)
#
#         save_pickled_file(shows_tags_filename,self._shows_tags_dict)
#
#     @staticmethod
#     def get_tag_index(tag_name, show_tags_list):
#         for i, tag in enumerate(show_tags_list):
#             if tag_name == tag['name']:
#                 return i
#         return -1
#
#     @staticmethod
#     def adjust_tag_percentage(tag):
#         p = tag['percentage']
#         if p >= 85:
#             adjusted_p = round((tag['percentage'] / 100), 3)
#         elif p >= 70:
#             second_coeff = 20 / 3 - tag['percentage'] / 15  # Can be a number from 1 to 1.5
#             adjusted_p = round((tag['percentage'] / 100) ** second_coeff, 3)
#             # 85 ---> 1, 70 ---> 2, 15 forward means -1 so m=-1/15, y-2 = -1/15(x-70) --> y=-1/15x + 20/3
#         elif p >= 60:
#             second_coeff = 16 - tag['percentage'] / 5
#             adjusted_p = round((tag['percentage'] / 100) ** second_coeff, 3)
#         else:
#             adjusted_p = 0
#         return adjusted_p
#
#     @staticmethod
#     def load_tags_database():
#         print(f"Unpacking part 1 of database")
#         tag_df = pl.read_parquet(f"{tags_db_filename}-P1.parquet")
#         for i in range(2, 1000):
#             try:
#                 print(f"Unpacking part {i} of database")
#                 temp_df = pl.read_parquet(f"{tags_db_filename}-P{i}.parquet")
#                 tag_df = tag_df.vstack(temp_df)
#                 # os.remove(f"{tags_db_filename}-P{i}.parquet")
#             except FileNotFoundError:
#                 break
#         return tag_df
#
#
# class AffinityDB:
#
#     _instance = None
#
#     def __new__(cls, *args, **kwargs):
#         """The class is a Singleton - we only need one instance of it since its purpose is
#         to house and create on demand all the data structures that are used in this project."""
#         if cls._instance is None:
#             cls._instance = super().__new__(cls, *args, **kwargs)
#         return cls._instance
#
#     def __init__(self):
#         # All properties are loaded on demand
#         self._df=None
#         self.anime_df = AnimeDB()
#         self.tags = Tags()
#
#     @property
#     def df(self):
#         """Polars database (stored as .parquet), contains username
#         + mean score + scored shows + all the scores of each user in it."""
#         if not isinstance(self._df, pl.DataFrame):
#             try:
#                 print("Loading affinity database")
#                 self._df = pl.read_parquet(f"{tags_db_filename}.parquet")
#             except FileNotFoundError:
#                 print("Affinity database not found. Creating new affinity database")
#                 self.create_affinity_DB()
#                 self._df = pl.read_parquet(f"{tags_db_filename}.parquet")
#         return self._df
#
#     @staticmethod
#     def create_affinity_DB():
#
#         def calculate_affinities_of_user(user_index):
#
#             nonlocal user_tag_affinity_dict
#             nonlocal database_dict
#
#             save_data_per = 10  # Each batch of N users will be separated into their own mini-database during
#             # runtime to avoid blowing up my poor 32GB of RAM
#             show_count = 0
#
#             user_name = partial_main_df['Username'][user_index]
#             user_scores = partial_main_df.select(relevant_shows).row(user_index, named=True)
#             watched_user_score_list = [score for title, score in user_scores.items() if score]
#             # ^ Simply a list of the user's scores for every show they watched
#
#             watched_titles = [title for title, score in user_scores.items() if score]
#             watched_MAL_score_dict = mean_score_row.select(watched_titles).to_dict(as_series=False)
#             watched_MAL_score_list = [score[0] for title, score in
#                                       watched_MAL_score_dict.items() if title != "Rows"]
#             # ^ Simply a list of the MAL scores for every show the user has watched
#
#             MAL_mean_of_watched = np.mean(watched_MAL_score_list)
#             user_mean_of_watched = np.mean(watched_user_score_list)
#             user_std_of_watched = np.std(watched_user_score_list)
#
#             user_tag_affinity_dict[user_name] = {tag_name: 0 for tag_name in tags.all_tags_list}
#             # ^ A dictionary that will store, for each user, their affinity to each of the 300+ tags we have
#
#             user_tag_counts = {tag_name: 0 for tag_name in tags.all_tags_list}
#             # ^ This dictionary will store the WEIGHTED (by each tag's percentage) tag count for each tag.
#
#             MAL_score_per_tag = {tag_name: 0 for tag_name in tags.all_tags_list}
#             user_score_per_tag = {tag_name: 0 for tag_name in tags.all_tags_list}
#             # ^ The above two are for calculating the average scores per tag later (both user and MAL averages)
#
#             tag_show_list = {tag_name: [] for tag_name in tags.all_tags_list}
#             # ^ A dictionary that will store, for each tag, all the shows that have said tag.
#
#             user_show_list = []
#
#             # Now, we will loop over every show that the user has watched
#             for show, user_score in user_scores.items():
#                 if user_scores[show]:
#
#                     user_show_list.append(show)
#                     MAL_score = mean_score_row[show].item()
#
#                     if MAL_score < 6.5:
#                         continue
#                         # We don't want to base decisions on a person's affinity to certain types of shows
#                         # on shows that are widely considered bad - for example, a romance fan won't necessarily
#                         # like a bad romance show more than someone indifferent towards romance shows.
#
#                     show_count += 1
#                     MAL_score_coeff = (-3 / 8) * MAL_score + 31 / 8
#
#                     show_tags_list = tags.shows_tags_dict[show]['Tags']
#                     show_genres_list = tags.shows_tags_dict[show]['Genres']
#                     show_studio = tags.shows_tags_dict[show]['Studio']
#
#                     for tag in show_tags_list:
#                         if tag['name'] not in tags.all_tags_list:
#                             continue
#                         adjusted_p = tags.adjust_tag_percentage(tag)
#                         # On Anilist, every tag a show has is listed with a percentage. This percentage
#                         # can be pushed down or up a bit by any registered user, so as a variable it
#                         # inevitably introduces human error - a tag with 60% means that not all users agree
#                         # that said tag is strongly relevant to the show, and thus we want to reduce its weight
#                         # by more than just 40% compared to a 100% tag.
#                         if adjusted_p == 0:
#                             break
#
#                         user_score_per_tag[tag['name']] += user_score * adjusted_p
#                         MAL_score_per_tag[tag['name']] += MAL_score * adjusted_p
#                         user_tag_counts[tag['name']] += adjusted_p
#                         tag_show_list[tag['name']].append(show)
#
#                     for genre in show_genres_list:
#                         user_score_per_tag[genre] += user_score
#                         MAL_score_per_tag[genre] += MAL_score
#                         user_tag_counts[genre] += 1
#                         # Genres and studios do not have percentages, so we add 1 as if p=100
#                         tag_show_list[genre].append(show)
#
#                     if show_studio in tags.all_tags_list:
#                         user_score_per_tag[show_studio] += user_score
#                         MAL_score_per_tag[show_studio] += MAL_score
#                         user_tag_counts[show_studio] += 1
#                         tag_show_list[show_studio].append(show)
#
#             # The above loop = func1? First put everything in classes tho
#
#             # After calculating the total scores for each tag, we calculate the "affinity" of the user
#             # to each tag.
#             for tag in tags.all_tags_list:
#                 try:
#                     tag_overall_ratio = user_tag_counts[tag] / show_count
#                     freq_coeff = min(1, max(user_tag_counts[tag] / 10, tag_overall_ratio * 20))
#                     # User has to watch either at least 10 shows with the tag or have the tag in at least 5% of their watched
#                     # shows for it to count fully.
#                     user_tag_diff = user_score_per_tag[tag] / user_tag_counts[tag] - user_mean_of_watched
#                     MAL_tag_diff = MAL_score_per_tag[tag] / user_tag_counts[tag] - MAL_mean_of_watched
#                     user_tag_affinity_dict[user_name][tag] = (2 * user_tag_diff - MAL_tag_diff) * freq_coeff
#
#                 except ZeroDivisionError:
#                     user_tag_affinity_dict[user_name][tag] = 0
#
#             user_tag_affinity_dict[user_name]['Shows'] = user_show_list
#
#             # Whatever is above - func2 (create_user_tag_affinity_dict? also draw it in the comments)
#             # Now we choose 20 random shows from the user's list. Each show will give us an entry in the final
#             # database. Each entry will consist of the user's affinities to each genre, their
#             random.shuffle(user_show_list)
#             user_shows_sample = user_show_list[0:20]
#
#             for show in user_shows_sample:
#                 show_tags_list = tags.shows_tags_dict[show]['Tags']
#                 show_genres_list = tags.shows_tags_dict[show]['Genres']
#                 show_studio = tags.shows_tags_dict[show]['Studio']
#
#                 # On Anilist, every show has other shows that are recommended by users. The "rating"
#                 # is how many users recommended it. We want to create a "Recommended Shows Affinity" column
#                 # while taking the weighted average of the user's affinities to the recommended shows.
#                 # The weight of each show is it's rating/total rating of all recommended shows.
#
#                 total_rec_rating = sum([rating for title, rating in tags.shows_tags_dict[show]['Recommended'].items()])
#                 recommended_shows = tags.shows_tags_dict[show]['Recommended']
#                 rec_affinity = 0
#
#                 for rec_anime, rec_rating in recommended_shows.items():
#                     if rec_anime in relevant_shows and user_scores[rec_anime] and rec_rating > 0:
#                         MAL_score = mean_score_row[rec_anime].item()
#                         MAL_score_coeff = (-3 / 8) * MAL_score + 31 / 8
#                         user_diff = user_scores[rec_anime] - user_mean_of_watched
#                         MAL_diff = watched_MAL_score_dict[rec_anime][0] - MAL_mean_of_watched
#                         rec_affinity += (user_diff - MAL_diff * MAL_score_coeff) * rec_rating
#
#                 try:
#                     weighted_aff = rec_affinity / total_rec_rating
#                     if np.isnan(weighted_aff) or np.isinf(weighted_aff):
#                         raise ValueError
#                     database_dict['Recommended Shows Affinity'].append(weighted_aff)
#                 except (ZeroDivisionError, ValueError) as e:  # No relevant recommended shows
#                     database_dict['Recommended Shows Affinity'].append(0)
#
#                 for tag in tags.all_tags_list:
#                     index = tags.get_tag_index(tag, show_tags_list)
#                     if index != -1:
#                         adjusted_p = tags.adjust_tag_percentage(show_tags_list[index])
#                         database_dict[tag].append(adjusted_p)
#                     else:
#                         if tag not in show_genres_list and tag != show_studio:
#                             database_dict[tag].append(0)
#                     database_dict[f"{tag} Affinity"].append(user_tag_affinity_dict[user_name][tag])
#
#                 for genre in show_genres_list:
#                     database_dict[genre].append(1)
#
#                 if show_studio in tags.all_tags_list:
#                     database_dict[show_studio].append(1)
#
#                 database_dict['User Score'].append(user_scores[show])
#                 database_dict['Show Score'].append(mean_score_row[show].item())
#                 database_dict['Mean Score'].append(user_mean_of_watched)
#                 database_dict['Standard Deviation'].append(user_std_of_watched)
#
#             if (user_index + 1) % save_data_per == 0:
#                 print(f"Finished processing user {user_index + 1}")
#                 save_pickled_file(f"user_tag_affinity_dict-P{(user_index + 1) // save_data_per}.pickle",
#                                   user_tag_affinity_dict)
#                 # Fix the above (either remove this entirely or concatenate them at the end?)
#
#                 for key in database_dict.keys():
#                     database_dict[key] = np.array(database_dict[key], dtype=np.float32)
#
#                 pl.DataFrame(database_dict).write_parquet(
#                     f"{tags_db_filename}-P{(user_index + 1) // save_data_per}.parquet")
#
#                 # After saving the data, we need to reinitialize the dicts to avoid wasting memory
#                 for tag in tags.all_tags_list:
#                     database_dict[tag] = []
#                     database_dict[f"{tag} Affinity"] = []
#
#                 database_dict = database_dict | {"Recommended Shows Affinity": [],
#                                                  "Show Score": [], "Mean Score": [], "Standard Deviation": [],
#                                                  "User Score": []}
#                 user_tag_affinity_dict = {}
#
#         user_db = UserDB()
#         anime_db = AnimeDB()
#         tags = Tags()
#         relevant_shows = list(tags.shows_tags_dict.keys())
#         # The shows in our tags dict are the ones filtered when creating it
#         # ( >= 15 min in length, >= 2 min per ep)
#
#         partial_main_df = user_db.df.select(user_db.stats + relevant_shows)
#         partial_anime_df = anime_db.df.select(["Rows"] + relevant_shows)
#
#         mean_score_row = partial_anime_df.filter(pl.col('Rows') == "Mean Score")
#         user_amount = partial_main_df.shape[0]
#
#         database_dict = {}
#         for tag in tags.all_tags_list:
#             database_dict[tag] = []
#             database_dict[f"{tag} Affinity"] = []
#         database_dict = database_dict | {"Recommended Shows Affinity": [],
#                                          "Show Score": [], "Mean Score": [], "Standard Deviation": [], "User Score": []}
#
#         # try:
#         #     # user_tag_affinity_dict = load_pickled_file("user_tag_affinity_dict3.pickle")
#         #     # #Change this
#         #     tags_db = pl.read_parquet("UserTagsDB.parquet")
#
#         # except FileNotFoundError:
#         for user_index in range(user_amount):
#             if user_index % 100 == 0:
#                 print(f"Currently on user {user_index}")
#             user_tag_affinity_dict = {}
#             calculate_affinities_of_user(user_index)
#
#         tags_db = tags.load_tags_database()
#         remove_zero_columns(tags_db)
#         tags_db.write_parquet(f"{tags_db_filename}.parquet")
#



# ------------------------------------------------------------------------------------------------------------
class Data:
    """ The main class the entire project works with. It is completely autonomous
    thanks to properties - functions outside this class can simply access any field
    within the class as if it exists without adding any extra logic, the class will
    take care of creating said field (database/list/dictionary) if it does not yet exist."""

    _instance=None

    # unlisted_titles = ['Dragon Ball Z: The Real 4-D',
    #                    'Seishun Buta Yarou wa Bunny Girl Senpai no Yume wo Minai Picture Drama',
    #                    'Douluo Dalu: Hao Tian Yang Wei',
    #                    'Douluo Dalu: Qian Hua Xi Jin',
    #                    'Douluo Dalu: Xiaowu Juebie',
    #                    'Douluo Dalu: Xingdou Xian Ji Pian',
    #                    'Gintama: Nanigoto mo Saisho ga Kanjin nanode Tashou Senobisuru Kurai ga Choudoyoi']


    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # All properties are loaded on demand
        self._main_df=None
        self._anime_df=None
        self._MAL_user_list = None
        self._blacklist = None
        self._scores_dict = None
        self._titles = None
        self._columns=None
        self._schema_dict=None

    @property
    def main_df(self):
        """Polars database (stored as .parquet), contains username
        + mean score + scored shows + all the scores of each user in it."""
        if not isinstance(self._main_df, pl.DataFrame):
            try:
                print("Loading main database")
                self._main_df = pl.read_parquet(user_database_name)
            except FileNotFoundError:
                print("Main database not found. Creating new main database")
                amount = int(input("Insert the desired amount of users\n"))
                self._main_df = pl.DataFrame(schema=self.schema_dict)
                self.fill_main_database(amount)
        return self._main_df

    @main_df.setter
    def main_df(self,value):
        """Not really necessary, here just in case"""
        if isinstance(value,pl.DataFrame):
            self._main_df = value
        else:
            raise ValueError("main_df must be a Polars database")


    # def non_sequel_main_df(self,value):
    #     if not isinstance(self._main_df, pl.DataFrame):
    #         try:
    #             print("Loading partial main database")
    #             relevant_titles = list(self.show_tags_dict.keys())
    #             self._partial_main_df = self.main_df.select(relevant_titles)

    @property
    def scores_dict(self):
        """A dictionary which holds the usernames of everyone in the database as keys,
        and their respective score arrays as values. Saved in a pickle file.
        Exists to significantly speed up real-time computation - working with the
        Polars database would require casting and slicing during computation."""
        if not self._scores_dict:
            print("Unpickling main scores dictionary")
            try:
                self._scores_dict = load_pickled_file(scores_dict_filename)
            except FileNotFoundError:
                print("Pickled dictionary not found. Checking for main database")
                if not isinstance(self._main_df, pl.DataFrame) or self._main_df.is_empty():
                    print("Main database not found, returning empty dictionary")
                    self._scores_dict = {}
                    # return self._scores_dict
                else:
                    print("Confirmed existence of main database. Converting to dictionary")
                    # This should never happen unless dict got manually deleted
                    self.__main_db_to_pickled_dict()
                    self._scores_dict = load_pickled_file(scores_dict_filename)
            except EOFError:
                # This should never happen after the debugging stage is over
                print("Pickle file is corrupted, converting main database.")
                self.__main_db_to_pickled_dict()
                self._scores_dict = load_pickled_file(scores_dict_filename)
        return self._scores_dict

    @scores_dict.setter
    def scores_dict(self, value):
        """Not really necessary, here just in case"""
        if isinstance(value, dict):
            self._scores_dict = value
        else:
            raise ValueError("scores_dict must be a dictionary")

    def save_scores_dict(self):
        """Utility function just to have 1 line instead of 2"""
        with open(scores_dict_filename, 'wb') as file:
            pickle.dump(self.scores_dict, file)

    def __main_db_to_pickled_dict(self):
        """This function should never be used, it is only there for debug emergencies such as
        the pickle file getting corrupted. Will take **A LOT** of memory and time because the
        transpose function will turn our uint8 db into float64/utf8."""
        score_df = self.main_df.select(self.titles)
        usernames = self.main_df["Username"]
        scores_dict = score_df.transpose(column_names=usernames).to_dict(as_series=False)
        self._scores_dict = {key: list_to_uint8_array(value) for key, value in scores_dict.items()}
        with open(scores_dict_filename, 'wb') as file:
            pickle.dump(self.scores_dict, file)

    @property
    def anime_df(self):
        """Database with info on all the anime that's listed (and ranked) on MAL.
        Fields are as shown in the "required_fields" variable of generate_anime_DB."""
        if not isinstance(self._anime_df, pl.DataFrame):
            try:
                print("Loading anime database")
                self._anime_df = pl.read_parquet(anime_database_name)
                print("Anime database loaded successfully")
            except FileNotFoundError:
                print("Anime database not found. Creating new anime database")
                self.generate_anime_DB()
                self._anime_df = pl.read_parquet(anime_database_name)
        return self._anime_df

    @anime_df.setter
    def anime_df(self, value):
        """Not really necessary, here just in case"""
        if isinstance(value, pl.DataFrame):
            self._anime_df = value
        else:
            raise ValueError("anime_df must be a Polars database")

    main_db_stats = ["Index", "Username", "Mean Score", "Scored Shows"]

    anime_db_stats = {'ID' : 0, 'Mean Score' : 1, 'Scores' : 2, 'Members' : 3, 'Episodes' : 4,
                      'Duration' : 5, 'Type' : 6, 'Year' : 7, 'Season' : 8}

    def get_stats_of_shows(self,show_list, relevant_stats):
        """ Will create a dictionary that has every show in show_list as the key, and every stat in relevant_stats
            in a list as the value.
            Example :
            {'Shingeki no Kyojin': {'ID': 16498.0, 'Mean Score': 8.53, 'Members': 3717089.0},
             'Shingeki no Kyojin OVA': {'ID': 18397.0, 'Mean Score': 7.87, 'Members': 439454.0}"""

        stats_dict = {}
        for show in show_list:
            show_dict = {}
            for stat in relevant_stats:
                try:
                    show_dict[stat] = self.anime_df.filter(pl.col('Rows') == stat)[show].item()
                except ColumnNotFoundError:
                    break
                except ValueError:
                    show_dict[stat] = None
            if show_dict:
                stats_dict[show] = show_dict
        return stats_dict


    def generate_anime_DB(self,non_sequels_only=False):
        """Creates the anime database which contains data (as defined in required_fields)
        on every show"""
        """""┌─────┬────────────┬────────────┬────────────┬───┬──────────┬────────────┬────────────┬────────────┐
        │ Row ┆ Fullmetal  ┆ Shingeki   ┆ "Oshi no   ┆ … ┆ Kokuhaku ┆ Hametsu no ┆ Utsu       ┆ Tenkuu     │
        │ s   ┆ Alchemist: ┆ no Kyojin: ┆ Ko"        ┆   ┆ ---      ┆ Mars       ┆ Musume     ┆ Danzai Ske │
        │ --- ┆ Brotherhoo ┆ The Final  ┆ ---        ┆   ┆ f64      ┆ ---        ┆ Sayuri     ┆ lter+Heave │
        │ str ┆ d          ┆ Se…        ┆ f64        ┆   ┆          ┆ f64        ┆ ---        ┆ n          │
        │     ┆ ---        ┆ ---        ┆            ┆   ┆          ┆            ┆ f64        ┆ ---        │
        │     ┆ f64        ┆ f64        ┆            ┆   ┆          ┆            ┆            ┆ f64        │
        ╞═════╪════════════╪════════════╪════════════╪═══╪══════════╪════════════╪════════════╪════════════╡
        │ ID  ┆ 5114.0     ┆ 51535.0    ┆ 52034.0    ┆ … ┆ 31634.0  ┆ 413.0      ┆ 13405.0    ┆ 3287.0     │
        │ Mea ┆ 9.1        ┆ 9.08       ┆ 9.08       ┆ … ┆ 2.3      ┆ 2.22       ┆ 1.98       ┆ 1.84       │
        │ n   ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ Sco ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ re  ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ Sco ┆ 2.004693e6 ┆ 140026.0   ┆ 134283.0   ┆ … ┆ 4847.0   ┆ 47441.0    ┆ 15790.0    ┆ 26203.0    │
        │ res ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ Mem ┆ 3.151508e6 ┆ 409744.0   ┆ 404237.0   ┆ … ┆ 6602.0   ┆ 65349.0    ┆ 20649.0    ┆ 37022.0    │
        │ ber ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ s   ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ Fav ┆ 216247.0   ┆ 8386.0     ┆ 12470.0    ┆ … ┆ 17.0     ┆ 297.0      ┆ 49.0       ┆ 115.0      │
        │ ori ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ tes ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ Epi ┆ 64.0       ┆ 2.0        ┆ 11.0       ┆ … ┆ 1.0      ┆ 1.0        ┆ 1.0        ┆ 1.0        │
        │ sod ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ es  ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ Dur ┆ 24.0       ┆ 61.0       ┆ 20.0       ┆ … ┆ 1.0      ┆ 19.0       ┆ 3.0        ┆ 19.0       │
        │ ati ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ on  ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ Typ ┆ 1.0        ┆ 4.0        ┆ 1.0        ┆ … ┆ 5.0      ┆ 3.0        ┆ 3.0        ┆ 3.0        │
        │ e   ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ Yea ┆ 2009.0     ┆ 2023.0     ┆ 2023.0     ┆ … ┆ 2015.0   ┆ 2005.0     ┆ 2003.0     ┆ 2004.0     │
        │ r   ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        │ Sea ┆ 2.0        ┆ 1.0        ┆ 2.0        ┆ … ┆ 3.0      ┆ 3.0        ┆ 1.0        ┆ 4.0        │
        │ son ┆            ┆            ┆            ┆   ┆          ┆            ┆            ┆            │
        └─────┴────────────┴────────────┴────────────┴───┴──────────┴────────────┴────────────┴────────────┘

        """
        def create_anime_DB_entry(anime, required_fields):
            # Helper function, creates a list which will later serve as a column in the anime DB.

            # def get_anime_duration(anime):
            #     print(f"Currently on anime {anime['title']}")
            #     text_duration = anime["duration"]
            #     if not text_duration:
            #         return None
            #     if 'sec' in text_duration:
            #         return 1
            #     text_duration_split = text_duration.split()
            #     if 'hr' in text_duration: # text_duration_split will be something like "1", "hr", "10", "min"
            #         try:
            #             return int(text_duration_split[0])*60 + int(text_duration_split[2])
            #         except IndexError:  # This means the duration is "2 hr" or "1 hr" rather than "1 hr 15 min".
            #             return int(text_duration_split[0])*60
            #     else: #
            #         return int(text_duration_split[0])

            def parse_start_date(date_str):
                try:
                    date_parsed = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    return date_parsed.year, (date_parsed.month - 1) // 3 + 1
                except ValueError:
                    try:
                        date_parsed = datetime.datetime.strptime(date_str, "%Y-%m")
                        return date_parsed.year, (date_parsed.month - 1) // 3 + 1
                    except ValueError:
                        return int(date_str), 1 # Only the year is known, season is assumed to be Winter (placeholder)


            # anime_data = list((itemgetter(*required_fields))(anime))
            anime_data = []
            for field in fields_no_edit_needed: # This takes care of the fields that can be put into db as they are
                anime_data.append(anime["node"][field])

            # The rest of the fields need editing or error handling
            if anime["node"]["num_episodes"]:
                anime_data.append(anime["node"]["num_episodes"])
            else: # This means episode_num = null, can only happen with VERY obscure shows.
                anime_data.append(1)

            if anime["node"]["average_episode_duration"]:
                anime_data.append(round(anime["node"]["average_episode_duration"]/60,1))
            else: # Same as above
                anime_data.append(20)

            # try:
            #     anime_data.append(get_anime_duration(anime)) # Duration needs separate handling
            # except ValueError:
            #     anime_data.append(20)  # Duration not listed (most likely very obscure show), setting default.

            try:
                media_type_index = MediaTypes[anime['node']["media_type"]].value
            except KeyError:
                media_type_index = None
            anime_data.append(media_type_index)

            try:
                year = int(anime["node"]["start_season"]["year"])
                season = anime["node"]["start_season"]["season"]
                season = Seasons[season].value # We convert the season to a numerical value for the database
            except KeyError:
                try:
                    year,season = parse_start_date(anime["node"]["start_date"])
                except KeyError:
                    year, season = None, None
            anime_data.append(year)
            anime_data.append(season)

            # print(anime["type"])
            # anime_data.append(anime["aired"]["prop"]["from"]['year'])
            # month = anime["aired"]["prop"]["from"]['month']
            # try:
            #     # season = Seasons((month - 1) // 3).name
            #     season = (month - 1) // 3 + 1
            # except TypeError:
            #     season = None  # Sometimes the season isn't listed
            # anime_data.append(season)

            # title = anime_data[0]
            # anime_data_dict[title] = anime_data[1:]

            title = anime["node"]["title"]
            anime_data_dict[title] = anime_data
            return title  # Title is returned to check whether we reached the last show

        def synchronize_main_dbs():
            # missing_cols = [x for x in self.titles if x not in self.main_df.columns]
            # for col_name in missing_cols:
            #     self.main_df = self.main_df.with_columns(
            #         pl.Series(col_name, [None] * len(self.main_df), dtype=pl.UInt8))

            missing_cols = [x for x in self.titles if x not in self.main_df.columns]
            for col_name in missing_cols:
                self.main_df = self.main_df.with_columns(
                    pl.Series(col_name, [None] * len(self.main_df), dtype=pl.UInt8))
            self.main_df.write_parquet(user_database_name)

        # def synchronize_dfs(df1,df2,cols1=None,cols2=None):
        #     # Add all columns in db2 but not in db1 to db1
        #     if cols1 and cols2:
        #         missing_cols = [x for x in cols2 if x not in cols1]
        #     else:
        #         missing_cols = [x for x in df2.columns if x not in df1.columns]
        #
        #     for col_name in missing_cols:
        #         df1 = df1.with_columns(
        #             pl.Series(col_name, [None] * len(df1), dtype=pl.UInt8))
        #
        #     return df1
            # self.main_df.write_parquet(main_database_name)
        # synchronize_dfs()
            # Also remove all titles that are in main but not in anime

        last_show_reached = False
        last_show = 'Tenkuu Danzai'
        # Lowest rated show, it'll be the last one in the sorted list.
        # Hardcoded because I'm 100% sure nothing will ever be rated
        # lower than THAT in my lifetime. The order is a bit off using Jikan
        # so checking for N/A score and stopping is not an option.

        required_fields = ["title", "mal_id", "score",
                           "scored_by", "members", "favorites"]

        url_required_fields = ["id","mean","num_scoring_users","num_list_users","num_episodes", "average_episode_duration",
                           "media_type","start_season", "start_date"]

        fields_no_edit_needed = ["id", "mean", "num_scoring_users", "num_list_users"]

        # The fields we need from the JSON object containing information about a single anime.

        # anime_data_dict = {'Rows': ['ID', 'Mean Score', 'Scores',
        #                             'Members', 'Favorites', 'Episodes', 'Duration', 'Type',  'Year', 'Season']}
        #
        anime_data_dict = {'Rows': ['ID', 'Mean Score', 'Scores',
                                    'Members', 'Episodes', 'Duration', 'Type',  'Year', 'Season']}

        page_num = 0
        while not last_show_reached:
            # We loop over pages of anime info that we get from the Jikan API (the info is
            # sorted by score, from highest to lowest) until we reach the lowest rated show.
            # There should be 25 shows per page/batch.

            anime_batch = get_anime_batch_from_MAL(page_num, url_required_fields)
            # url = f'https://api.jikan.moe/v4/top/anime?page={page_num}'
            try:
                print(f"Currently on score {anime_batch['data'][-1]['node']['mean']}")
            except KeyError:
                print("Finished")
            while anime_batch is None:
                # If we failed to get the batch for some reason, we retry until it's a success.
                # Jikan API does not require authorization, so the only reason for a failure
                # could be an outage in the API itself, in which case we wouldn't want to
                # timeout/stop the function as an outage could technically last for a few hours,
                # or even days.
                print("Error - unable to get batch. Sleeping just to be safe, "
                      "then trying again.")
                logging.error("Error - unable to get batch. Sleeping just to be safe, "
                              "then trying again.")
                time.sleep(Sleep.LONG_SLEEP)
                # anime_batch = call_function_through_process(get_search_results, url)
                anime_batch = get_anime_batch_from_MAL(page_num, url_required_fields)
                print(anime_batch)

            for anime in anime_batch["data"]:
                if not non_sequels_only or anime_is_valid(anime):
                    title = create_anime_DB_entry(anime, url_required_fields)
                if title.startswith(last_show):
                    last_show_reached = True
                    break
            page_num += 1
        table = pa.Table.from_pydict(anime_data_dict)
        pq.write_table(table, anime_database_name) # This creates a .parquet file from the dict
        if self._main_df:
            synchronize_main_dbs()

    @property
    def MAL_users_list(self):
        """CSV list of all the usernames in the database. Not strictly necessary anymore,
        currently serves as backup."""
        if not self._MAL_user_list:
            print("Loading users list")
            try:
                with open(MAL_users_filename, newline="", encoding='utf-8') as f:
                    self._MAL_user_list = next(csv.reader(f, delimiter=','))
                    print(f"{MAL_users_filename} loaded successfully")
                    usernames = list(self.main_df['Username'])
                    if len(self._MAL_user_list) != len(usernames):
                        print("User list data was not up-to-date, returning updated user list")
                        save_list_to_csv(usernames, MAL_users_filename)
                        return usernames
                    return self._MAL_user_list
            except FileNotFoundError as e:
                print("Unable to load list - CSV file not found. Returning empty list")
                self._MAL_user_list = []
        return self._MAL_user_list

    @property
    def blacklist(self):
        """List of all users that didn't meet the requirements to have their lists
        in the database. Stored here to avoid wasting an API request on checking
        their lists again."""
        if not self._blacklist:
            print("Loading blacklist")
            try:
                with open(blacklist_users_filename, newline="", encoding='utf-8') as f:
                    self._blacklist = next(csv.reader(f, delimiter=','))
                    print(f"{blacklist_users_filename} loaded successfully")
                    return self._blacklist
            except FileNotFoundError as e:
                print("Unable to load blacklist - file not found. Returning empty list")
                self._blacklist = []
        return self._blacklist

    @property #change this into a normal var?
    def schema_dict(self):
        """The type schema of the main Polars database."""
        if not self._schema_dict:
            self._schema_dict = {'Index' : pl.Int64, 'Username': pl.Utf8, 'Mean Score': pl.Float32, 'Scored Shows': pl.UInt32} | \
                                {x: y for (x, y) in zip(self.titles, [pl.UInt8] * len(self.titles))}
        return self._schema_dict


    @property
    def columns(self):
        """Columns of the main database"""
        if not self._columns:
            self._columns = ['Index','Username','Mean Score', 'Scored Shows'] + self.titles
        return self._columns

    @property
    def titles(self): #change this into a normal var?
        """A list of all the anime titles."""
        if not self._titles:
            self._titles = self.anime_df.columns[1:] # anime_df will automatically be generated
            # for title in self._titles:
            #     if title in self.unlisted_titles:
            #         self._titles.remove(title)
            # through the property if it doesn't exist yet
        return self._titles

    def save_main_df(self):
        """Used during the creation of the main database. Saves all relevant files
        (main database, user list, blacklist and scores dictionary) every N created
        entries as defined in fill_main_database."""
        self._main_df.write_parquet(user_database_name)
        usernames = list(self._main_df['Username'])
        print(f"Saving MAL user list. Length is {len(usernames)}")
        save_list_to_csv(usernames,MAL_users_filename)
        print(f"Saving blacklist. Length is {len(self._blacklist)}")
        save_list_to_csv(self._blacklist,blacklist_users_filename)
        print(f"Saving scores dictionary")
        self.save_scores_dict()
        print("Finished saving")


    @timeit
    def fill_main_database(self,amount):
        """ This function adds users to the main database. If empty, it will create a new one.
        The "amount" parameter is the final desired user count. """

        @timeit
        def add_user_list_to_db_list(user_index, username, user_list):
            """Takes a single user's list and creates a row for them in the main database
            if user meets criteria (>50 scored shows, >30 days account_age, non-troll mean score).
            returns True if user was added, False otherwise."""

            current_time = datetime.datetime.now(datetime.timezone.utc)
            account_age_threshold = datetime.timedelta(days=30)
            min_scored_shows = 50  # User needs to have scored at least 50 shows to be part of the DB
            user_scored_shows = count_scored_shows(user_list)

            if user_scored_shows >= min_scored_shows:
                show_amount = 0
                score_sum = 0
                account_age_verified = False
                for anime in user_list:
                    title = anime['node']['title']
                    score = anime['list_status']['score']

                    if score == 0:
                        actual_user_index = user_index + initial_users + saved_so_far
                        print(f'Finished processing {show_amount} shows of user {username} ({actual_user_index})')
                        break

                    else:
                        if not account_age_verified:
                            update_timestamp = anime['list_status']['updated_at']
                            time_since_update = current_time - datetime.datetime.fromisoformat(update_timestamp)
                            if time_since_update > account_age_threshold:
                                account_age_verified = True
                        # We test whether the account is at least one month old by seeing if at least one
                        # anime update was done more than a month ago.
                        show_amount += 1

                        if title in self.titles:
                            scores_db_dict[title][user_index] = score
                        score_sum += score

                mean_score = round(score_sum / show_amount, 4)

                if 2 <= mean_score <= 9.7:  # First we filter by scores, < 2 and > 9.7 means will just clutter the data
                    if not account_age_verified:  # If we couldn't verify through the anime list, we check the
                        # user's page directly. We only want to use this as a last resort since unlike the
                        # former it takes another API call to do so.
                        account_age = check_account_age_directly(username)
                        if account_age < account_age_threshold:
                            print(f"{username}'s account is {account_age} old, likely a troll")
                            return False
                        print(f"{username}'s account is {account_age} old, user verified. Adding to database")
                    scores_db_dict['Index'][user_index] = current_users
                    scores_db_dict['Username'][user_index] = username
                    scores_db_dict['Scored Shows'][user_index] = show_amount
                    scores_db_dict['Mean Score'][user_index] = mean_score
                    new_user_list = list_to_uint8_array([scores_db_dict[key][user_index]
                                                         for key in self.titles])
                    self.scores_dict[username] = new_user_list
                    return True
                else:
                    print(f"{username} has no meaningful scores, proceeding to next user")
                    return False
            print(f"{username} has less than 50 scored shows, proceeding to next user")
            return False

        def save_data():

            print(f"Saving database. Currently on {current_users} entries")
            logger.debug(f"Saving database. Currently on {current_users} entries")

            # First we create a df from the temporary dictionary. Then we concatenate
            # it with the existing database.
            nonlocal saved_so_far
            temp_df = pl.DataFrame(scores_db_dict, schema=self.schema_dict)
            saved_so_far += temp_df.shape[0]

            try:
                self._main_df = pl.concat(
                    [
                        self._main_df,
                        temp_df,
                    ],
                    how="vertical",
                )
            except ValueError:
                self._main_df = temp_df

            # After we concatenated the main and temp databases, we need to save all the
            # necessary files (polars db, user list + blacklist and scores dict)
            # to avoid losing data in case the program stops for whatever reason.
            self.save_main_df()

        save_data_per = 100
        saved_so_far = 0
        min_scored_amount = 75000
        max_scored_amount = 500000

        # We want to take usernames from shows that are not TOO popular, but also not too niche.
        # The reason for this is that the update table of a niche show will include many troll accounts
        # that basically either put everything in their list, or are score-boosting alts with many shows.
        # The update table of a very popular show like Attack on Titan on the other hand, will include many
        # people that have just started watching anime, and thus only have a few shows in their list + score-boosting
        # alts for those shows specifically. Since the program runs for weeks, we want to avoid wasting time
        # on filtering those as much as possible.

        @timeit
        def initialize_temp_scores_dict():
            """ We use a dictionary to avoid filling the db inplace, which takes MUCH more time. The data
            is saved every save_data_per entries, after which the dictionary is reset to None values to avoid
            it getting huge and taking up memory."""
            remaining_amount = amount - current_users
            for key in self.columns:
                scores_db_dict[key] = [None] * min(save_data_per, remaining_amount)

        # ------------------------------ Main function starts here ----------------------------

        scores_db_dict = {}
        """This dictionary exists as temporary storage for user lists. Instead creating a new polars row
        and adding it to the database as we get it, we add it to this dictionary. Then, each
        save_data_per entries, we convert that dictionary to a Polars dataframe and concatenate
        it with the main one. This results in significant speedup."""

        current_users = len(self.MAL_users_list)
        initial_users = current_users

        rows = self.anime_df.rows()
        ids = rows[0][1:]
        scored_amount = rows[2][1:]

        initialize_temp_scores_dict()

        print(f"Starting MAL user length/database size is : {len(self.MAL_users_list)}")
        print(f"Starting Blacklist length is : {len(self.blacklist)}")

        ids_titles_scored = list(zip(ids, self.titles, scored_amount))
        shuffle(ids_titles_scored)  # Shuffle to start getting users from a random show, not the top shows
        # We start by iterating over the shows we have in our show DB. Then, for each show, we get 375 usernames.
        while True:
            for id, title, scored_amount in ids_titles_scored:
                if not scored_amount or int(scored_amount) < min_scored_amount \
                                or int(scored_amount) > max_scored_amount:
                    print(f"Scored amount of {title} is {scored_amount}, moving to next show")
                    continue
                print(f"Scored amount of {title} is {scored_amount}, proceeding with current show")

                title = replace_characters_for_url(title)
                base_url = f"https://myanimelist.net/anime/{id}/{title}/stats?"
                print(base_url)
                users_table = get_usernames_from_show(base_url)
                # This returns a table of list updates

                for table_row in users_table:
                    # The list update table includes usernames, timestamps and more.
                    # We extract the usernames from there by their assigned CSS class.
                    if current_users == amount:
                        break

                    user_link = table_row.findNext(
                        "div", {"class": "di-tc va-m al pl4"}).findNext("a")
                    user_name = str(user_link.string) # This thing is actually a bs4.SoupString,
                    # not a regular Python string.

                    if user_name not in self.MAL_users_list and user_name not in self.blacklist:

                        if user_name.startswith('ishinashi'):
                            logger.debug("Retard troll detected, continuing on")
                            continue

                        user_anime_list = get_user_MAL_list(user_name, full_list=False)
                        dict_user_index = current_users - initial_users - saved_so_far
                        added = add_user_list_to_db_list(dict_user_index,
                                                         user_name, user_anime_list)

                        if added:
                            current_users += 1
                            self.MAL_users_list.append(user_name)
                            if (current_users - initial_users) % save_data_per == 0:
                                # We save the database and the lists every n users just to be safe
                                save_data()
                                if current_users == amount:
                                    return
                                initialize_temp_scores_dict()
                        else:
                            self.blacklist.append(user_name)
                    else:
                        print(f"{user_name} has already been added/blacklisted, moving on to next user")

                if current_users == amount:
                    save_data()
                    # We reached the necessary amount of users in the database
                    return

    def main_df_user_indexes(self):
        pass


    def create_user_entry(self,user_name):
        """Creates a user entry for the one we're calculating affinity for + returns a Polars row of their stats.
        Currently not in use due to being time-inefficient at adding in real-time. New process maybe?"""
        if user_name not in self.MAL_users_list and user_name not in self.blacklist:
            user_list = get_user_MAL_list(user_name, full_list=False)
            if not user_list:
                print("Unable to access private user list. Please make your list public and try again.")
                terminate_program()
        else:
            print("User already exists in database")
            return

        old_titles = [x for x in self.main_df.columns if x not in self.main_db_stats]

        anime_indexes = {v: k for (k, v) in enumerate(old_titles)}
        # new_list = [None] * len(self.titles)
        new_list = [None] * (len(old_titles))

        show_amount = 0
        score_sum = 0
        for anime in user_list:  # anime_list, scores_db
            title = anime['node']['title']
            score = anime['list_status']['score']
            if score == 0:
                break
            show_amount += 1
            if title in self.titles:
                new_list[anime_indexes[title]] = score
            score_sum += score

        mean_score = round(score_sum / show_amount, 4)
        new_list_for_return = list_to_uint8_array(new_list)  # We save the original list here,
        # because later we'll have to add fields that are relevant for adding to DB but not for list comparison.

        if 2 <= mean_score <= 9.7:  # Removing troll accounts/people who score everything 10/10
            index = self.main_df.shape[0]
            new_list = [index,user_name,mean_score,show_amount] + new_list

            schema_dict = {'Index' : pl.Int64, 'Username': pl.Utf8, 'Mean Score': pl.Float32, 'Scored Shows': pl.UInt32} | \
                          {x: y for (x, y) in zip(old_titles, [pl.UInt8] * len(old_titles))}
            # cols = ['Index', 'Username', 'Mean Score', 'Scored Shows'] + self.titles

            new_dict = {k : v for k,v in zip(self.main_df.columns, new_list)}

            new_row = pl.DataFrame(new_dict,schema=schema_dict)
            print(new_row)

            # new_row = synchronize_dfs(new_row,self.main_df)
            self.main_df = pl.concat(
                [
                    self.main_df,
                    new_row,
                ],
                how="vertical",
            )

            self.main_df.write_parquet(user_database_name)
            self.MAL_users_list.append(user_name)
            save_list_to_csv(self.MAL_users_list,MAL_users_filename)
        return new_list_for_return





def replace_characters_for_url(s):
    """This function replaces characters in the titles of the shows as they are
    returned by the MAL API,
    according to the way they're supposed to be written in the URL. There doesn't
    seem to be any consistency -
    some characters like "-" are retained as they are, while others are omitted or
    replaced with an underscore.
    Seriously, exclamation marks are omitted while question marks sometimes
    are...and sometimes aren't."""

    replace_with_empty = [",", "(", ")", "'", ".", "!", "\"", "?", "[", "]", "{", "}",
                          "$", "#", "@", "%", "^", "*"]  #
    replace_with_underscore = [" ", ":", "&", ";", "/", "+"]
    for ch in replace_with_empty:
        s = s.replace(ch, "")
    for ch in replace_with_underscore:
        s = s.replace(ch, "_")
    return s


def get_usernames_from_show(base_url):
    """This function takes an anime title, accesses the updates table on the Stats
    tab on the MAL page of that title, and returns pages_to_get*75 usernames from it.
    This is the easiest way to gather recently active usernames."""
    pages_to_get = 10
    users_table = []
    too_many_requests_flag = 0
    page_num = 0
    while page_num < pages_to_get:
        page_users_table = []
        url = f"{base_url}show={str(75 * page_num)}"
        print(f"Getting usernames from {url}")
        logger.debug(f"Getting usernames from {url}")
        page_html = call_function_through_process(get_search_results, url)
        try:
            soup = BeautifulSoup(page_html.text, "html.parser")
            update_table = soup.find("table", {"class": "table-recently-updated"})
            page_users_table = update_table.findAll('tr')[1:]
            too_many_requests_flag = 0
        # flag gets reset to 0 if succeeded in getting data from soup
        except AttributeError:
            if too_many_requests_flag:
                # flag being 1 means we got an error twice in a row,
                # most likely an issue with too many requests
                print("Too many requests, commencing sleep then trying again")
                logger.debug("Too many requests, commencing sleep then trying again")
                time.sleep(Sleep.LONG_SLEEP)
            else:
                base_url = f"{base_url}m=all&"
                page_num = page_num - 1
            too_many_requests_flag = 1
        users_table = users_table + page_users_table
        page_num = page_num + 1
    print(f'Length of the table returned from get_usernames_from_show'
          f' : {len(users_table)}')  #
    return users_table


def allowed_type_but_invalid_show(anime, relation_type):
    """Helper function for filtering out shows that already have a related entry
       in the database (only for used for non-sequels-only database).

        Parameters :

            anime - the JSON object representing a single show. See MAL API
            documentation for more info.

            relation_type -

        """
    match relation_type:
        case "parent_story" | "alternative_version":
            # This case mainly filters one-off OVAs of shows that eventually got a full
            # adaptation (but the OVA's ID will obviously be lower). Note that if the
            # alternative version of a show is a TV show,
            # it is most likely a remake - and thus should count as a valid show (for
            # example Fruits Basket 2001 vs 2019).
            if anime["media_type"] != 'tv':
                return True
        case "other":
            relation_types = [related_anime["relation_type"] for related_anime in
                              anime["related_anime"]]  #
            if anime["media_type"] == 'special' or \
                    (anime["media_type"] not in 'tvona' and 'prequelsequel' in relation_types):  #
                return True
            # If the anime is a "special" or is something that isn't a TV/ONA
            # (basically a regular series), but does have a prequel/sequel - it
            # gets filtered.
    return False


def anime_is_valid(anime) -> bool:  # TURN THIS INTO A DATABASE
    """This function is used for filtering out shows that already have a related entry
        in the database (only used for non-sequels-only database). For example, season 2
        of any show will be considered invalid, as will any special episodes that have a
        separate entry on MAL.

        Parameters :
            anime_id : the MAL ID of the anime to be checked

        Returns :
            True/False

    """

    def add_entry(title, valid):
        with open("ValidTitles.csv", "w", encoding='utf-8', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([title, valid])

    try:
        valid_titles = pd.read_csv("ValidTitles.csv")
        try:
            anime_row = valid_titles.loc[valid_titles['Title'] == anime['title']]
            return anime_row['Valid'].bool()
        except KeyError:
            pass  # If the entry for the anime isn't in the database, it's probably
            # a new show that hasn't been put in yet. The function will proceed
            # to add it to the database and return the appropriate T/F value.
    except FileNotFoundError:
        pass  # If file not found, function proceeds to create it from scratch.
        # There should never be a need to do this

    url = f'https://api.myanimelist.net/v2/anime/{anime["mal_id"]}?fields=id,media_type,mean,' \
          f'related_anime,genres,' \
          f'average_episode_duration,num_episodes,num_scoring_users'  #
    # This url will return more detailed information about the anime we're checking
    # than we have in the anime object passed to the function

    anime = call_function_through_process(get_search_results, url)
    # If we got an error, keep sleeping and trying until it works (or until
    # get_search_results terminates automatically in case of critical error)
    while anime is None:
        logger.warning("Anime was returned as None, sleeping and retrying")
        time.sleep(Sleep.LONG_SLEEP)  # Just in case
        print("Anime was returned as None, sleeping and retrying")
        anime = call_function_through_process(get_search_results, url)
    # except (urllib.error.HTTPError, requests.HTTPError,
    # http.client.RemoteDisconnected):
    #     print("---------- Connection was reset, waiting 60 seconds ----------")
    #     time.sleep(60)
    #     return anime_is_valid(anime_id)
    print(Fore.LIGHTBLUE_EX + anime["title"])

    allowed_types = ["alternative_setting", "parent_story", "other", "character",
                     "alternative_version"]  #
    anime_duration_seconds = anime["num_episodes"] * anime["average_episode_duration"]

    if (anime_duration_seconds < 1800 and anime["media_type"] != 'movie') \
            or anime_duration_seconds < 600:
        print(Fore.RED + f'INVALID - Anime duration is only {anime_duration_seconds}')
        add_entry(anime["title"], False)
        return False  # This disqualifies any OVAs (or really short movies).
    if anime["num_scoring_users"] < 2000:  #
        print(Fore.RED + f'INVALID - Only {anime["num_scoring_users"]} scored the anime')
        add_entry(anime["title"], False)
        return False  # This disqualifies shows with too little data to be meaningful
    for related_anime in anime["related_anime"]:

        # For every show the anime that's being tested has a relationship with,
        # first of all we check whether the ID of the related anime is smaller (meaning
        # something related to the tested anime already existed before it).
        # If it is, we then check whether despite that, the anime being tested is a
        # standalone. This could happen with several relationship types as described in
        # "allowed_types".
        # Afterwards, we do a third check through filter_allowed_types,
        # which does a separate case check for each relation type on whether the show
        # truly is a standalone.

        relation_type = related_anime["relation_type"]
        if (related_anime["node"]["id"] < anime["id"] and (
                relation_type not in allowed_types or  #
                allowed_type_but_invalid_show(anime, relation_type))):  #
            print(Fore.RED + 'INVALID - Related anime already found in list')
            add_entry(anime["title"], False)
            return False

    print(Fore.LIGHTGREEN_EX + f'VALID - Duration of the show above is '
                               f'{anime["num""_episodes"] * anime["average_episode_duration"]}'
                               f' and {anime["num_scoring_users"]} gave it a score.')  #
    add_entry(anime["title"], True)
    return True


def check_account_age_directly(user_name):
    """ Used in case we couldn't verify that the user's account is older than 30 days
    through their anime list (could be an Anilist migrator). In this case, we have to make
    an extra API call to make sure the user wasn't created in the last 30 days."""
    url = f'https://myanimelist.net/profile/{user_name}'
    page_html = call_function_through_process(get_search_results, url)
    if not page_html:
        page_html = call_function_through_process(get_search_results, url)
        if not page_html:
            return datetime.datetime.now(datetime.timezone.utc) # Something wrong with the page html, very rare
    soup = BeautifulSoup(page_html.text, "html.parser")
    date_str = soup.findAll("span", {"class": "user-status-data di-ib fl-r"})[-1].text #We'll always need the last one
    date_obj = datetime.datetime.strptime(date_str, "%b %d, %Y")
    account_creation_time = date_obj.replace(tzinfo=datetime.timezone.utc)
    current_time = datetime.datetime.now(datetime.timezone.utc)
    time_since_account_creation = current_time - account_creation_time
    return time_since_account_creation


def get_anime_batch_from_jikan(page_num):
    """Utility function to get a batch of 25 shows from the Jikan API. Only exists because
    it sometimes throws weird errors which require us to retry so that we don't skip any shows"""
    url = f'https://api.jikan.moe/v4/top/anime?page={page_num}'
    anime_batch = call_function_through_process(get_search_results, url)
    while anime_batch is None:
        # We keep retrying until we get the batch, the URL is correct so the main reason
        # for failure would be the API itself being down, in which case we just wait for it
        # to come back up.
        print("Anime batch was returned as None, sleeping and retrying")
        logger.debug("Anime batch was returned as None, sleeping and retrying")
        time.sleep(Sleep.MEDIUM_SLEEP)
        anime_batch = call_function_through_process(get_search_results, url)
    return anime_batch


def get_anime_batch_from_MAL(page_num, required_fields):

    url = f'https://api.myanimelist.net/v2/anime/ranking?ranking_type=all&limit=100&offset={page_num*100}&' \
          'fields='
    for field in required_fields:
        url = url + field + ","  # Concatenate all required fields with a comma inbetween
    url = url[:-1] # Remove the last comma
    anime_batch = call_function_through_process(get_search_results, url)
    while anime_batch is None:
        # We keep retrying until we get the batch, the URL is correct so the main reason
        # for failure would be the API itself being down, in which case we just wait for it
        # to come back up.
        print("Anime batch was returned as None, sleeping and retrying")
        logger.debug("Anime batch was returned as None, sleeping and retrying")
        time.sleep(Sleep.MEDIUM_SLEEP)
        anime_batch = call_function_through_process(get_search_results, url)
    return anime_batch



def count_scored_shows(user_list):
    count = 0
    for anime in user_list:
        count += 1
        if anime['list_status']['score'] == 0:
            return count
    return count


@timeit
def get_user_MAL_list(user_name, full_list=True):
    """Helper function of fill_list_database. Gets the full list of one MAL user via their username.
    If full_list is false, it'll stop once it gets to the shows with no score"""
    url = f'https://api.myanimelist.net/v2/users/' \
          f'{user_name}/animelist?fields=list_status&limit=1000&sort=list_score&nsfw=True'

    response = call_function_through_process(get_search_results, url)
    try:
        anime_list = response["data"]
    except (TypeError, KeyError) as ex:
        anime_list = []  # list is empty/private
        print(response)
        print(ex)

    # If the user has more than 1000 entries in their list, we will need separate API
    # calls for each thousand.

    if len(anime_list) == 1000:
        scored_shows = count_scored_shows(anime_list)
        thousands = 1
        while len(anime_list) == 1000 * thousands and (scored_shows == 1000 * thousands
                                                       or full_list):
            print(
                f'Length of {user_name}s anime list exceeds {1000 * thousands}, '  #
                f'proceeding to next 1000 shows')
            logger.debug(
                f'Length of {user_name}s anime list exceeds {1000 * thousands}, '
                f'proceeding to next 1000 shows')
            url = f'https://api.myanimelist.net/v2/users/' \
                  f'{user_name}/animelist?fields=list_status&limit=1000&sort' \
                  f'=list_score' \
                  f'&offset={1000 * thousands}&nsfw=True'  #
            response = get_search_results(url)
            next_part = response["data"]
            anime_list = anime_list + next_part
            thousands += 1
            scored_shows = count_scored_shows(anime_list)

    return anime_list

