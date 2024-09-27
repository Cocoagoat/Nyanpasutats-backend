import datetime
from random import shuffle

try:
    import thread
except ImportError:
    import _thread as thread

import os
from main.modules.filenames import *
from main.modules.MAL_utils import MALUtils
from main.modules.AnimeDB import AnimeDB
from main.modules.GlobalValues import cpu_share
from main.modules.Errors import UserListFetchError
from main.modules.AnimeListHandler import AnimeListHandler, MALListHandler
import polars as pl
import pickle
import csv
from main.modules.general_utils import load_pickled_file, list_to_uint8_array, save_list_to_csv, save_pickled_file


class UserDB:
    """This class creates UserDB.parquet and holds all relevant functions and information
    regarding it. Current fields (columns) :

    - Index
    - Username
    - Mean Score
    - Scored Shows
    - One column for each show that exists as a column in AnimeDB.

    An example database is shown in the class definition itself.
    """
    _instance = None

#     "───────┬────────────┬──────────┬────────────┬───┬──────────┬────────────┬────────────┬────────────┐
# │ Index ┆ Username   ┆ Mean     ┆ Scored     ┆ … ┆ Kokuhaku ┆ Hametsu no ┆ Utsu       ┆ Tenkuu     │
# │ ---   ┆ ---        ┆ Score    ┆ Shows      ┆   ┆ ---      ┆ Mars       ┆ Musume     ┆ Danzai Ske │
# │ i64   ┆ str        ┆ ---      ┆ ---        ┆   ┆ u8       ┆ ---        ┆ Sayuri     ┆ lter+Heave │
# │       ┆            ┆ f32      ┆ u32        ┆   ┆          ┆ u8         ┆ ---        ┆ n          │
# │       ┆            ┆          ┆            ┆   ┆          ┆            ┆ u8         ┆ ---        │
# │       ┆            ┆          ┆            ┆   ┆          ┆            ┆            ┆ u8         │
# ╞═══════╪════════════╪══════════╪════════════╪═══╪══════════╪════════════╪════════════╪════════════╡
# │ 0     ┆ XXXXXXXXXX ┆ 8.6158   ┆ 177        ┆ … ┆ null     ┆ null       ┆ null       ┆ null       │
# │ 1     ┆ XXXXXXXXX  ┆ 7.6709   ┆ 158        ┆ … ┆ null     ┆ null       ┆ null       ┆ null       │
# │ 2     ┆ XXXXXXXXXX ┆ 8.285    ┆ 207        ┆ … ┆ null     ┆ null       ┆ null       ┆ null       │
# │       ┆            ┆          ┆            ┆   ┆          ┆            ┆            ┆            │
# │ 3     ┆ XXXXXXXXX  ┆ 8.4023   ┆ 87         ┆ … ┆ null     ┆ null       ┆ null       ┆ null       │
# │ …     ┆ …          ┆ …        ┆ …          ┆ … ┆ …        ┆ …          ┆ …          ┆ …          │
# │ 16    ┆ XXXXXXXXXX ┆ 7.885    ┆ 113        ┆ … ┆ null     ┆ null       ┆ null       ┆ null       │
# │       ┆            ┆          ┆            ┆   ┆          ┆            ┆            ┆            │
# │ 17    ┆ XXXXXXX    ┆ 8.0755   ┆ 53         ┆ … ┆ null     ┆ null       ┆ null       ┆ null       │
# │ 18    ┆ XXXXXXXXXX ┆ 8.8571   ┆ 112        ┆ … ┆ null     ┆ null       ┆ null       ┆ null       │
# │       ┆            ┆          ┆            ┆   ┆          ┆            ┆            ┆            │
# │ 19    ┆ XXXXXXXXXX ┆ 7.6111   ┆ 54         ┆ … ┆ null     ┆ null       ┆ null       ┆ null       │
# │       ┆            ┆          ┆            ┆   ┆          ┆            ┆            ┆            │
# └───────┴────────────┴──────────┴────────────┴───┴──────────┴────────────┴────────────┴────────────┘"
# (Actual amount of users in the database should be 150,000)
    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._df = None
            cls._instance._MAL_user_list = None
            cls._instance._blacklist = None
            cls._instance._scores_dict = None
            cls._instance._columns = None
            cls._instance._schema_dict = None
            cls._instance.anime_db = AnimeDB()
            cls._user_amount = 0
            # cls.continue_filling = kwargs.get('continue_filling', False)
        return cls._instance

    def __init__(self, continue_filling=False):
        self.continue_filling = continue_filling
        # All properties are loaded on demand

    stats = ["Index", "Username", "Mean Score", "Scored Shows"]

    @property
    def df(self):

        # def user_prompted_filling():
        #     # return (input("Continue filling database? Y/N") == 'Y')
        #     return True

        """Polars database (stored as .parquet), contains username
        + mean score + scored shows + all the scores of each user in it."""
        if not isinstance(self._df, pl.DataFrame):
            file_loaded = False

            try:
                print("Loading user database")
                self._df = pl.read_parquet(user_database_name)
                file_loaded = True

            except FileNotFoundError:
                print("User database not found. Creating new user database")
                self._df = pl.DataFrame(schema=self.schema_dict)

            # continue_filling = not file_loaded or self.continue_filling
            if self.continue_filling or not file_loaded:
                amount = int(input("Insert the desired amount of users\n"))
                self._user_amount = amount
                self.fill_main_database(amount)

        self._user_amount = len(self._df)
        return self._df

    @df.setter
    def df(self, value):
        """Not really necessary, here just in case"""
        if isinstance(value, pl.DataFrame):
            self._df = value
        else:
            raise ValueError("df must be a Polars database")

    def split_df(self, parts):
        print("Beginning to split main database")
        df_size = len(self.df)
        part_size = int(df_size/parts)

        partials_folder_path = Path(f'{str(user_database_name.parent).split(".")[0]}\\Partials')
        if not os.path.exists(partials_folder_path):
            os.mkdir(partials_folder_path)

        for i in range(parts):
            print(f"Currently on part {i+1}")
            if i != parts-1:
                df_part = self.df[i*part_size: (i+1)*part_size]
            else:
                df_part = self.df[i*part_size:df_size]  # In case df_size/parts was rounded down by the casting
            df_part.write_parquet(partials_folder_path
                                  / f'{str(user_database_name.name).split(".")[0]}-P{i+1}.parquet')

    def get_df_part(self, i):
        try:
            print(f"Loading part {i} of user database")
            return pl.read_parquet(f'{str(user_database_name.parent).split(".")[0]}\\Partials'
                                   f'\\{str(user_database_name.name).split(".")[0]}-P{i}.parquet')
        except FileNotFoundError:
            print(f"Part {i} not found, splitting user db")
            parts = int(os.cpu_count() / cpu_share)
            self.split_df(parts)
            return pl.read_parquet(f'{str(user_database_name.parent).split(".")[0]}\\Partials'
                                   f'\\{str(user_database_name.name).split(".")[0]}-P{i}.parquet')

    @property
    def scores_dict(self):
        """A dictionary which holds the usernames of everyone in the database as keys,
        and their respective score arrays as uint8 values. Saved in a pickle file.
        Exists to significantly speed up real-time computation - working with the
        Polars database would require casting and slicing during computation."""

        "{'XXXXXX': array([0, 0, 0, ..., 0, 0, 0], dtype=uint8)," \
        " 'XXXXXX': array([10,  0, 10, ...,  0,  0,  0], dtype=uint8)," \
        " 'XXXXXX': array([7, 8, 0, ..., 0, 0, 0], dtype=uint8)"

        if not self._scores_dict:
            print("Unpickling main scores dictionary")
            try:
                self._scores_dict = load_pickled_file(scores_dict_filepath)
            except FileNotFoundError:
                print("Pickled dictionary not found. Checking for main database")
                if not isinstance(self._df, pl.DataFrame) or self._df.is_empty():
                    print("Main database not found, returning empty dictionary")
                    self._scores_dict = {}
                else:
                    print("Confirmed existence of main database. Converting to dictionary")
                    # This should never happen unless dict got manually deleted
                    self.__main_db_to_pickled_dict()
                    self._scores_dict = load_pickled_file(scores_dict_filepath)
            except EOFError:
                # This should never happen after the debugging stage is over
                print("Pickle file is corrupted, converting main database.")
                self.__main_db_to_pickled_dict()
                self._scores_dict = load_pickled_file(scores_dict_filepath)
        return self._scores_dict

    @scores_dict.setter
    def scores_dict(self, value):
        if isinstance(value, dict):
            self._scores_dict = value
        else:
            raise ValueError("scores_dict must be a dictionary")

    def split_scores_dict(self):
        split_full_path = str(scores_dict_filepath).split("\\")
        path, filename = '\\'.join(split_full_path[:-1]), split_full_path[-1].split(".")[0]
        for i in range(100):
            size = round(len(self.scores_dict)/100)
            dict_part_i = dict(list(self.scores_dict.items())[i*size:(i+1)*size])
            save_pickled_file(Path(path) / "Partials" / f"{filename}-P{i+1}.pickle", dict_part_i)

    def save_scores_dict(self):
        with open(scores_dict_filepath, 'wb') as file:
            pickle.dump(self.scores_dict, file)

    def __main_db_to_pickled_dict(self):
        """This function should never be used, it is only there for debug emergencies such as
        the pickle file getting corrupted. Will take **A LOT** of memory and time because the
        transpose function will turn our uint8 db into float64/utf8."""
        score_df = self.main_df.select(self.titles)
        usernames = self.main_df["Username"]
        scores_dict = score_df.transpose(column_names=usernames).to_dict(as_series=False)
        self._scores_dict = {key: list_to_uint8_array(value) for key, value in scores_dict.items()}
        with open(scores_dict_filepath, 'wb') as file:
            pickle.dump(self.scores_dict, file)

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
                    usernames = list(self.df['Username'])
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
        their lists again when their usernames are encountered."""
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

    @property  # change this into a normal var?
    def schema_dict(self):
        """The type schema of the main Polars database."""
        if not self._schema_dict:
            self._schema_dict = {'Index': pl.Int64, 'Username': pl.Utf8, 'Mean Score': pl.Float32,
                                 'Scored Shows': pl.UInt32} | \
                                {x: y for (x, y) in zip(self.anime_db.titles, [pl.UInt8] * len(self.anime_db.titles))}
        return self._schema_dict

    @property
    def columns(self):
        """Columns of the main database"""
        if not self._columns:
            self._columns = ['Index', 'Username', 'Mean Score', 'Scored Shows'] + self.anime_db.titles
        return self._columns

    def save_df(self):
        """Used during the creation of the main database. Saves all relevant files
        (main database, user list, blacklist and scores dictionary) every N created
        entries as defined in fill_main_database."""

        usernames = list(self._df['Username'])
        print(len(usernames), self._user_amount)
        print(f"Saving MAL user list. Length is {len(usernames)}")
        self._df.write_parquet(user_database_name)
        save_list_to_csv(usernames, MAL_users_filename)
        print(f"Saving blacklist. Length is {len(self._blacklist)}")
        save_list_to_csv(self._blacklist, blacklist_users_filename)
        print(f"Saving scores dictionary")

        self.save_scores_dict()

        if len(usernames) == self._user_amount:
            print("Splitting scores dictionary")
            self.split_scores_dict()
        print("Finished saving")

    # @timeit
    def fill_main_database(self, amount):
        """ This function adds users to the main database. If empty, it will create a new one.
        The "amount" parameter is the final desired user count. """

        # @timeit
        def add_user_list_to_db_list(user_index, username, user_list):
            """Takes a single user's list and creates a row for them in the main database
            if user meets criteria (>50 scored shows, >30 days account_age, non-troll mean score).
            returns True if user was added, False otherwise."""

            def remove_last_user_scores():
                for v in scores_db_dict.values():
                    v[user_index] = None

            current_time = datetime.datetime.now(datetime.timezone.utc)
            account_age_threshold = datetime.timedelta(days=30)
            min_scored_shows = 50  # User needs to have scored at least 50 shows to be part of the DB
            user_scored_shows = MALUtils.count_scored_shows(user_list)

            if user_scored_shows >= min_scored_shows:
                show_amount = 0
                score_sum = 0
                account_age_verified = False
                for anime in user_list.list_obj:
                    title = anime['node']['title']
                    score = anime['list_status']['score']

                    if score == 0:
                        actual_user_index = user_index + initial_users + saved_so_far
                        print(f'Finished processing {show_amount} shows of user {username} ({actual_user_index})')
                        if show_amount < 50:
                            print(f"{username} had less than 50 relevant scored entries, proceeding to next user")
                            remove_last_user_scores()
                            return False
                        break

                    else:
                        if not account_age_verified:
                            update_timestamp = anime['list_status']['updated_at']
                            time_since_update = current_time - datetime.datetime.fromisoformat(update_timestamp)
                            if time_since_update > account_age_threshold:
                                account_age_verified = True
                        # We test whether the account is at least one month old by seeing if at least one
                        # anime update was done more than a month ago.
                        # show_amount += 1

                        if title in self.anime_db.titles:
                            scores_db_dict[title][user_index] = score
                            show_amount += 1
                            score_sum += score
                        # score_sum += score

                mean_score = round(score_sum / show_amount, 4)

                if 2 <= mean_score <= 9.7:  # First we filter by scores, < 2 and > 9.7 means will just clutter the data
                    if not account_age_verified:  # If we couldn't verify through the anime list, we check the
                        # user's page directly. We only want to use this as a last resort since unlike the
                        # former it takes another API call to do so.
                        account_age = MALUtils.check_account_age_directly(username)
                        if account_age < account_age_threshold:
                            print(f"{username}'s account is {account_age} old, likely a troll")
                            remove_last_user_scores()
                            return False
                        print(f"{username}'s account is {account_age} old, user verified. Adding to database")
                    scores_db_dict['Index'][user_index] = current_users
                    scores_db_dict['Username'][user_index] = username
                    scores_db_dict['Scored Shows'][user_index] = show_amount
                    scores_db_dict['Mean Score'][user_index] = mean_score
                    new_user_list = list_to_uint8_array([scores_db_dict[key][user_index]
                                                         for key in self.anime_db.titles])
                    self.scores_dict[username] = new_user_list
                    return True
                else:
                    print(f"{username} has no meaningful scores, proceeding to next user")
                    remove_last_user_scores()
                    return False
            print(f"{username} has less than 50 scored shows, proceeding to next user")
            return False

        def save_data():

            print(f"Saving database. Currently on {current_users} entries")

            # First we create a df from the temporary dictionary. Then we concatenate
            # it with the existing database.
            nonlocal saved_so_far
            temp_df = pl.DataFrame(scores_db_dict, schema=self.schema_dict)
            saved_so_far += temp_df.shape[0]

            try:
                self._df = pl.concat(
                    [
                        self._df,
                        temp_df,
                    ],
                    how="vertical",
                )
            except ValueError:
                self._df = temp_df

            # After we concatenated the main and temp databases, we need to save all the
            # necessary files (polars db, user list + blacklist and scores dict)
            # to avoid losing data in case the program stops for whatever reason.
            self.save_df()

        save_data_per = 100
        saved_so_far = 0
        min_scored_amount = 20000
        max_scored_amount = 60000

        # We want to take usernames from shows that are not TOO popular, but also not too niche.
        # The reason for this is that the update table of a niche show will include many troll accounts
        # that basically either put everything in their list, or are score-boosting alts with many shows.
        # The update table of a very popular show like Attack on Titan on the other hand, will include many
        # people that have just started watching anime, and thus only have a few shows in their list + score-boosting
        # alts for those shows specifically. Since the program runs for days, we want to avoid wasting time
        # on filtering those as much as possible.

        # @timeit
        def initialize_temp_scores_dict():
            """ We use a dictionary to avoid filling the db inplace, which takes MUCH more time. The data
            is saved every save_data_per entries, after which the dictionary is reset to None values to avoid
            it getting huge and taking up memory."""
            remaining_amount = amount - current_users
            for key in self.columns:
                scores_db_dict[key] = [None] * min(save_data_per, remaining_amount)

        # ------------------------------ Main function starts here ----------------------------

        scores_db_dict = {}
        """This dictionary exists as temporary storage for user lists. Instead of creating a new
         polars row and adding it to the database as we get it, we add it to this dictionary. 
         Then, each save_data_per entries, we convert that dictionary to a Polars dataframe and
         concatenate it with the main one. This results in significant speedup."""

        current_users = len(self.MAL_users_list)

        initial_users = current_users

        rows = self.anime_db.df.rows()
        ids = rows[0][1:]
        scored_amount = rows[2][1:]

        initialize_temp_scores_dict()

        print(f"Starting MAL user length/database size is : {len(self.MAL_users_list)}")
        print(f"Starting Blacklist length is : {len(self.blacklist)}")

        ids_titles_scored = list(zip(ids, self.anime_db.titles, scored_amount))
        shuffle(ids_titles_scored)  # Shuffle to start getting users from a random show, not the top shows
        # We start by iterating over the shows we have in our show DB. Then, for each show, we get 375 usernames.
        while True:
            for id, title, scored_amount in ids_titles_scored:
                if not scored_amount or int(scored_amount) < min_scored_amount \
                        or int(scored_amount) > max_scored_amount:
                    print(f"Scored amount of {title} is {scored_amount}, moving to next show")
                    continue
                print(f"Scored amount of {title} is {scored_amount}, proceeding with current show")

                title = MALUtils.replace_characters_for_url(title)
                base_url = f"https://myanimelist.net/anime/{id}/{title}/stats?"
                print(base_url)
                users_table = MALUtils.get_usernames_from_show(base_url)
                # This returns a table of list updates

                for table_row in users_table:
                    # The list update table includes usernames, timestamps and more.
                    # We extract the usernames from there by their assigned CSS class.
                    if current_users == amount:
                        break

                    user_link = table_row.findNext(
                        "div", {"class": "di-tc va-m al pl4"}).findNext("a")
                    user_name = str(user_link.string)  # This thing is actually a bs4.SoupString,
                    # not a regular Python string.

                    if user_name not in self.MAL_users_list and user_name not in self.blacklist:

                        if user_name.startswith('ishinashi'):  # Anti-troll measures, remove later
                            continue
                        try:
                            list_handler = MALListHandler(user_name, full_list=False)
                            user_anime_list = list_handler.anime_list
                        except UserListFetchError:
                            continue  # Either user's list is private or doesn't exist, moving on to next username
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

    def get_user_db_entry(self, user_name, site="MAL"):
        titles = AnimeDB(anime_database_updated_name).titles
        ListHandler = AnimeListHandler.get_concrete_handler(site)
        list_handler = ListHandler(user_name)
        user_scores_list, mean_score, show_amount = list_handler.get_user_scores_list(db_row=True, for_predict=True)

        index = 0   # Index doesn't really matter, just needs to be there for the structure
        new_list = [index, user_name, mean_score, show_amount] + user_scores_list

        schema_dict = {'Index': pl.Int64, 'Username': pl.Utf8, 'Mean Score': pl.Float32, 'Scored Shows': pl.UInt32} | \
                      {x: y for (x, y) in zip(titles, [pl.UInt8] * len(titles))}

        new_dict = {k: v for k, v in zip(self.stats + titles, new_list)}

        new_row = pl.DataFrame(new_dict, schema=schema_dict)

        return new_row
