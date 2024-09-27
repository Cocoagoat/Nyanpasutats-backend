import os
from random import random
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from polars import ColumnNotFoundError
from tqdm import tqdm
from .UserDB import UserDB
from .Tags import Tags
from .User import User
from .Errors import UserListFetchError
from .GeneralData import GeneralData as GeneralData
from .UserAffinityCalculator import UserAffinityCalculator
from .filenames import *
from .Graphs import Graphs
from .GlobalValues import cpu_share
from concurrent.futures import ProcessPoolExecutor
import time
from main.modules.AnimeDB import AnimeDB
from dataclasses import dataclass, field
from .general_utils import load_pickled_file, timeit, save_pickled_file, time_at_current_point, shuffle_df, handle_nans, list_to_uint8_array, add_suffix_to_filename


class AffDBEntryCreator:

    def __init__(self, user: User, data: GeneralData, tags: Tags, for_predict=False):
        self.user = user
        self.data = data
        self.features = AffinityDB.Features()
        self.tags = tags
        mean_scores = AnimeDB(anime_database_updated_name if for_predict else None).mean_scores
        self.user_affinity_calculator = UserAffinityCalculator(self.user, self.data, self.tags, mean_scores,
                                                               use_updated_dbs=for_predict)
        self.aff_db_entry_dict = None
        self.graphs = Graphs()
        self.for_predict = for_predict

    @classmethod
    def initialize_aff_db_dict(cls, db_type=1, for_predict=False):
        """Creates dictionary with each of the database's columns pointing to an empty list
        to which entries will be appended in the future."""

        features = AffinityDB.Features()
        tags = Tags()
        if db_type == 1:
            aff_db_entry_dict = {stat: [] for stat in features.overall_tag_features}

            for category in list(tags.tags_per_category.keys()) + ["Genres"]:
                aff_db_entry_dict = aff_db_entry_dict | {f"{category} {' '.join(stat.split()[1:])}": []
                                                         for stat in features.category_features
                                                         if "Double" not in stat and "Doubles" not in category}
            aff_db_entry_dict = aff_db_entry_dict | {"Studio Affinity": [], "Studio Pos Affinity": []}

        elif db_type == 2:
            aff_db_entry_dict = {f"{tag} Affinity": [] for tag in tags.all_anilist_tags}
            aff_db_entry_dict = aff_db_entry_dict | {f"{tag} Pos Affinity": [] for tag in tags.all_anilist_tags}

        aff_db_entry_dict = aff_db_entry_dict | {stat: [] for stat in features.other_features}

        if not for_predict:
            del aff_db_entry_dict['Show Name']
        return aff_db_entry_dict

    def get_tag_weight(self, tag_name, entry_tags_list, entry_tag_percentages, use_adj=False):

        try:
            tag_index = entry_tags_list.index(tag_name)
        except ValueError:
            return 0, 0

        try:
            adjusted_p = entry_tag_percentages[tag_index]
        except IndexError:
            adjusted_p = 1

        if adjusted_p == 0:
            return 0, 0

        if not use_adj:
            dict_to_use = self.user.tag_affinity_dict
            pos_dict_to_use = self.user.tag_pos_affinity_dict
        else:
            dict_to_use = self.user.adj_tag_affinity_dict
            pos_dict_to_use = self.user.adj_pos_tag_affinity_dict

        try:
            aff_tag_weight = dict_to_use[tag_name] * adjusted_p
            pos_aff_tag_weight = pos_dict_to_use[tag_name] * adjusted_p  # check with create single user
        except (KeyError, TypeError) as e:
            return 0, 0

        return aff_tag_weight, pos_aff_tag_weight

    def get_affinity_stats(self, entry, category, category_tag_list, overall_temp_dict):
        """

        :param entry: remove
        :param category: Name of the tag category to be processed.
        :param category_tag_list: List of tags under the category.
        :param entry_tags_dict:
        :param entry_genres_list: remove
        :param overall_temp_dict: Dictionary that holds the current max/min value for each max/min/avg category.
        """

        def update_stats(stats_dict, tag_weight, aff_tag_weight, pos_aff_tag_weight, tag_type):

            if aff_tag_weight > 0:
                stats_dict[f"{tag_type} Pos Affinity Ratio"] += tag_weight
            elif aff_tag_weight < 0:
                stats_dict[f"{tag_type} Neg Affinity Ratio"] += tag_weight

            stats_dict[f"{tag_type} Avg Affinity"] += aff_tag_weight*tag_weight
            stats_dict[f"{tag_type} Avg Pos Affinity"] += pos_aff_tag_weight*tag_weight

            if stats_dict[f"{tag_type} Min Affinity"] is None or aff_tag_weight < stats_dict[
                f"{tag_type} Min Affinity"]:
                stats_dict[f"{tag_type} Min Affinity"] = aff_tag_weight

            if stats_dict[f"{tag_type} Max Affinity"] is None or aff_tag_weight > stats_dict[
                f"{tag_type} Max Affinity"]:
                stats_dict[f"{tag_type} Max Affinity"] = aff_tag_weight

            if stats_dict[f"{tag_type} Max Pos Affinity"] is None or pos_aff_tag_weight \
                    > stats_dict[f"{tag_type} Max Pos Affinity"]:
                stats_dict[f"{tag_type} Max Pos Affinity"] = pos_aff_tag_weight

        tag_type = self.tags.get_category_tag_type(category)  # Single or Double
        entry_tags_dict = self.tags.entry_tags_dict2[entry] if not self.for_predict else self.tags.entry_tags_dict2_updated[entry]
        temp_dict = self.initialize_temp_dict()

        #  We loop over each tag in the category, updating the temp_dict which represents the stats
        #  for said category after getting each tag's weight. (For example if temp_dict['Single Max Affinity']
        #  is 0.5, and the next tag 'Space' has an aff_tag_weight of 1.0, after update_stats
        #  temp_dict['Single Max Affinity'] will be 1.0.
        for tag_name in category_tag_list:
            try:
                tag_p = entry_tags_dict[tag_name]['percentage']
                aff_tag_weight = self.user.adj_tag_affinity_dict[tag_name] * tag_p
                pos_aff_tag_weight = self.user.adj_pos_tag_affinity_dict[tag_name] * tag_p
            except KeyError:
                # Tag not part of this entry
                continue
            if not aff_tag_weight:
                continue  # No point updating if the user has 0 affinity to that tag
            update_stats(temp_dict, entry_tags_dict[tag_name]['percentage'],
                         aff_tag_weight, pos_aff_tag_weight, tag_type)

        # At this point ratio = sum, it hasn't been divided by total yet
        tag_count = temp_dict[f"{tag_type} Pos Affinity Ratio"] + temp_dict[f"{tag_type} Neg Affinity Ratio"]

        # Averages = sums as well until this point, need to divide by tag count to get true averages
        try:
            temp_dict[f"{tag_type} Avg Affinity"] /= tag_count
            temp_dict[f"{tag_type} Avg Pos Affinity"] /= tag_count
        except ZeroDivisionError:
            temp_dict[f"{tag_type} Avg Affinity"] = 0
            temp_dict[f"{tag_type} Avg Pos Affinity"] = 0

        if category != 'Studio':
            for stat in self.features.category_features:  # "single/double max/min/avg/maxpos affinity"
                if category == 'Doubles':
                    # 'Doubles' is simply a category that allows us to loop over the doubles.
                    # It should not have separate "Doubles Max Affinity" categories in the main
                    # dict/database because we already cover that when updating f"{tag_type} Max Affinity"
                    # when looping over doubletags.
                    break
                if tag_type not in stat:
                    continue
                if temp_dict[stat] is None:
                    temp_dict[stat] = 0
                self.aff_db_entry_dict[f"{category} {' '.join(stat.split()[1:])}"].append(temp_dict[stat])

        elif temp_dict["Single Max Affinity"]:  # If the values in the dictionary are None, that means the studio isn't
            # one of the big studios we recognize as a tag (there are hundreds of studios, some of them only made
            # 2-3 shows so "affinity" to those studios would be meaningless)
            self.aff_db_entry_dict["Studio Affinity"].append(temp_dict["Single Max Affinity"])
            self.aff_db_entry_dict["Studio Pos Affinity"].append(temp_dict["Single Max Pos Affinity"])
        else:
            self.aff_db_entry_dict["Studio Affinity"].append(0)
            self.aff_db_entry_dict["Studio Pos Affinity"].append(0)

        for key in temp_dict.keys():
            if not temp_dict[key]:
                continue
            stat_name = " ".join(key.split()[1:])
            if stat_name.startswith("Avg"):
                # If key doesn't start with min or max, it's a counter (avg or neg/pos)
                overall_temp_dict[key] += temp_dict[key]*tag_count
                # Multiplying by tag count because later we divide by the sum of the tag counts to get the true avg
            elif stat_name.startswith("Min"):
                if overall_temp_dict[key] is None or temp_dict[key] < overall_temp_dict[key]:
                    overall_temp_dict[key] = temp_dict[key]
            elif stat_name.startswith("Max"):  # It's one of the "max" values
                if overall_temp_dict[key] is None or temp_dict[key] > overall_temp_dict[key]:
                    overall_temp_dict[key] = temp_dict[key]
            else:  # It's the pos/neg tag ratios
                overall_temp_dict[key] += temp_dict[key]

    @staticmethod
    def initialize_temp_dict():
        temp_dict = {"Max Affinity": None, "Min Affinity": None, "Avg Affinity": 0,
                     "Max Pos Affinity": None, "Avg Pos Affinity": 0, "Pos Affinity Ratio": 0,
                     "Neg Affinity Ratio": 0}  # Basic stats with no distinction between tag types

        temp_dict = {f"{tag_type} {stat}": stat_value for tag_type in Tags().tag_types
                     for stat, stat_value in temp_dict.items()}  # Quick way to create stats for each tag type
        return temp_dict

    def create_db_entries_from_user_list(self, shuffle_list=False,
                                         shows_to_take="watched", db_type=1, sample_size=0,
                                         site="MAL"):
        def get_recommended_show_aff():

            total_rec_rating = sum(
                [rating for title, rating in entry_tags_dict[entry]['Recommended'].items()])
            recommended_shows = entry_tags_dict[entry]['Recommended']
            rec_affinity = 0
            relevant_shows = list(entry_tags_dict.keys())
            for rec_anime, rec_rating in recommended_shows.items():
                if rec_anime in relevant_shows and self.user.scores[rec_anime] and rec_rating > 0:
                    MAL_score = mean_scores[rec_anime]
                    MAL_score_coeff = -0.6 * MAL_score + 5.9
                    user_diff = self.user.scores[rec_anime] - self.user.mean_of_watched
                    MAL_diff = mean_scores[rec_anime] - self.user.MAL_mean_of_watched
                    rec_affinity += (user_diff - MAL_diff * MAL_score_coeff) * rec_rating

            try:
                weighted_aff = rec_affinity / total_rec_rating
                if np.isnan(weighted_aff) or np.isinf(weighted_aff):
                    raise ValueError(f"Recommended shows affinity calculation error for {entry}")
                return weighted_aff
            except (ZeroDivisionError, ValueError):  # No relevant recommended shows
                return 0

        def get_user_entry_list(shows_to_take):
            relevant_shows = list(entry_tags_dict.keys())
            if shows_to_take == "watched":
                # This will always be used for calculating affinities to create the affinity database.
                entry_list = [key for key, value in self.user.scores.items() if value]
            elif shows_to_take == "watched+related":

                temp_list = []
                entry_list = [key for key, value in self.user.scores.items() if value]
                for entry in entry_list:
                    root, related_shows = graphs.find_related_entries(entry)
                    temp_list += related_shows
                entry_list = list(set(entry_list + temp_list))
            elif shows_to_take == "unwatched":
                # For predicting scores of shows the user hasn't watched yet.
                del self.aff_db_entry_dict['User Score']
                entry_list = [entry for entry in entry_tags_dict.keys()
                                        if entry in relevant_shows
                              and not self.user.scores[entry]]
            elif shows_to_take == "both":
                # Predicting on both unwatched and watched shows (to test performance)
                del self.aff_db_entry_dict['User Score']
                entry_list = [entry for entry in show_tags_dict.keys()
                                        if entry in relevant_shows]
            elif shows_to_take == "all":
                # For predicting scores of every single entry (rather than just every show).
                del self.aff_db_entry_dict['User Score']
                entry_list = relevant_shows
            else:
                raise ValueError("shows_to_predict expects a value of either"
                                 " 'watched', 'unwatched', 'both' or 'all'.")
            return entry_list

        self.aff_db_entry_dict = self.initialize_aff_db_dict(db_type, self.for_predict)

        self.user = self.user_affinity_calculator.get_user_affs()
        if not self.user:
            if self.for_predict:
                raise UserListFetchError("This user hasn't scored any relevant shows yet.")
            # Only possible if user has no relevant watched shows (e.g their list
            # is full of hentai/exclusively low scored shows)
            return

        if self.for_predict:
            anime_db_filename = anime_database_updated_name
            graphs = self.graphs.all_graphs_nls_updated
            entry_tags_dict = self.tags.entry_tags_dict_nls_updated
            entry_tags_dict2 = self.tags.entry_tags_dict2_updated
            show_tags_dict = self.tags.show_tags_dict_nls_updated
        else:
            anime_db_filename = anime_database_name
            graphs = self.graphs.all_graphs_nls
            entry_tags_dict = self.tags.entry_tags_dict_nls
            entry_tags_dict2 = self.tags.entry_tags_dict2
            show_tags_dict = self.tags.show_tags_dict_nls

        anime_db = AnimeDB(anime_db_filename)
        mean_scores = anime_db.mean_scores
        mean_scores_list = np.array(list(anime_db.mean_scores.values()))

        user_scores_list = np.array(list(self.user.scores.values()))
        user_scores_mask = user_scores_list!=None

        MAL_affinity = np.corrcoef(np.float64(user_scores_list[user_scores_mask]),
                                   mean_scores_list[user_scores_mask])[0][1]

        self.user.entry_list = get_user_entry_list(shows_to_take)
        if shuffle_list:
            random.shuffle(self.user.entry_list)
        if sample_size:
            self.user.entry_list = self.user.entry_list[0:sample_size]

        processed_entries = []
        for entry in self.user.entry_list:

            if entry in processed_entries:
                continue
            try:
                main_entry = entry_tags_dict[entry]['Main']
            except KeyError:
                continue
            self.user.adj_tag_affinity_dict = self.user.tag_affinity_dict.copy()
            self.user.adj_pos_tag_affinity_dict = self.user.tag_pos_affinity_dict.copy()

            self.user_affinity_calculator.recalc_affinities_2(main_entry)

            for entry, length_coeff in show_tags_dict[main_entry]['Related'].items():
                processed_entries.append(entry)
                if shows_to_take == 'watched' and not self.user.scores[entry]:
                    continue

                entry_tags_list = entry_tags_dict[entry]['Tags'] + entry_tags_dict[entry][
                    'DoubleTags']
                entry_genres_list = entry_tags_dict[entry]['Genres']
                entry_studio = entry_tags_dict[entry]['Studio']
                entry_tag_names = [tag['name'] for tag in entry_tags_list]
                entry_tag_percentages = [tag['percentage'] for tag in entry_tags_list]

                if entry == main_entry:
                    self.aff_db_entry_dict['Sequel'].append(0)
                    self.aff_db_entry_dict['Score Difference'].append(0)
                else:
                    try:
                        self.aff_db_entry_dict['Score Difference'].append(mean_scores[entry] - mean_scores[main_entry])
                    except (ColumnNotFoundError, KeyError):
                        continue  # A unique case of a sequel that meets the conditions of partial_anime_df +
                        # a main show that doesn't (for example sequel rated 6.8 but main show rated 6.1).
                        # We don't want to count this.
                    self.aff_db_entry_dict['Sequel'].append(1)

                self.aff_db_entry_dict['Recommended Shows Affinity'].append(get_recommended_show_aff())

                if self.for_predict:
                    self.aff_db_entry_dict['Show Name'].append(entry)

                if db_type == 1:
                    overall_temp_dict = self.initialize_temp_dict()

                    for category, category_tags in self.tags.tags_per_category.items():
                        self.get_affinity_stats(entry, category, category_tags,
                                                overall_temp_dict)

                    category = "Genres"
                    self.get_affinity_stats(entry, category, category_tag_list=entry_genres_list,
                                            overall_temp_dict=overall_temp_dict)

                    category = "Studio"
                    self.get_affinity_stats(entry, category, category_tag_list=[entry_studio],
                                            overall_temp_dict=overall_temp_dict)

                    for tag_type in Tags().tag_types:
                        overall_temp_dict[f"{tag_type} Tag Count"] = overall_temp_dict[f"{tag_type} Pos Affinity Ratio"]\
                                               + overall_temp_dict[f"{tag_type} Neg Affinity Ratio"]

                        try:
                            for stat in [f"{tag_type} Avg Affinity", f"{tag_type} Avg Pos Affinity",
                                         f"{tag_type} Pos Affinity Ratio", f"{tag_type} Neg Affinity Ratio"]:
                                overall_temp_dict[stat] /= overall_temp_dict[f"{tag_type} Tag Count"]
                        except ZeroDivisionError:
                            for stat in [f"{tag_type} Avg Affinity", f"{tag_type} Avg Pos Affinity",
                                         f"{tag_type} Pos Affinity Ratio", f"{tag_type} Neg Affinity Ratio"]:
                                overall_temp_dict[stat] = 0

                    for key in overall_temp_dict.keys():
                        self.aff_db_entry_dict[key].append(overall_temp_dict[key])

                # elif db_type == 2:  # Currently not in use, old db type
                #     entry_tag_names = entry_tag_names + entry_genres_list + [entry_studio]
                #
                #     if self.user.scores[entry] and self.for_predict:
                #         self.user_affinity_calculator.recalc_affinities_without_entry(entry)
                #         use_adj = True
                #     else:
                #         use_adj = False
                #
                #     for tag_name in self.tags.all_anilist_tags:
                #         aff_tag_weight, pos_aff_tag_weight = self.get_tag_weight(tag_name, entry_tag_names,
                #                                                                  entry_tag_percentages,
                #                                                                  use_adj=use_adj)
                #         self.aff_db_entry_dict[f"{tag_name} Affinity"].append(aff_tag_weight)
                #         self.aff_db_entry_dict[f"{tag_name} Pos Affinity"].append(pos_aff_tag_weight)

                if shows_to_take.startswith("watched"):
                     self.aff_db_entry_dict['User Score'].append(self.user.scores[entry])

                show_score = mean_scores[entry]
                self.aff_db_entry_dict['Show Score'].append(show_score)
                self.aff_db_entry_dict['Mean Score'].append(self.user.mean_of_watched)
                self.aff_db_entry_dict['MALAffinity'].append(MAL_affinity)
                self.aff_db_entry_dict['Standard Deviation'].append(self.user.std_of_watched)

                entry_members = anime_db.scored_amounts[entry]
                self.aff_db_entry_dict['Show Popularity'].append(entry_members)
                self.aff_db_entry_dict['User Scored Shows'].append(self.user.scored_amount)
                self.aff_db_entry_dict['Length Coeff'].append(
                    show_tags_dict[main_entry]['Related'][entry])

        for key in list(self.aff_db_entry_dict.keys()):
            if key.startswith('Doubles'):
                del self.aff_db_entry_dict[key]


#  ------------------------------------------------------------------------------------------------
class AffinityDB:
    _instance = None

    total_minor_parts = 3000
    size_limit = 20_000_000

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, db_type=None):
        self._df = None
        self._data = None
        self.features = self.Features()
        self._OG_aff_means = None
        self._major_parts = None
        self._db_type = db_type
        self.graphs = Graphs()

    @dataclass
    class Features:
        category_features: list = field(default_factory=
                                        lambda: [f"{tag_type} {feature}" for tag_type in Tags().tag_types
                                                   for feature in ["Max Affinity", "Min Affinity", "Avg Affinity",
                                                                   "Max Pos Affinity"]])

        overall_tag_features: list = field(default_factory=
                                           lambda: [f"{tag_type} {feature}" for tag_type in Tags().tag_types
                                                   for feature in ["Max Affinity", "Min Affinity", "Avg Affinity",
                                                                   "Max Pos Affinity", "Avg Pos Affinity",
                                                                   "Pos Affinity Ratio", "Neg Affinity Ratio",
                                                                   "Tag Count"]])
        other_features: list = field(
            default_factory=lambda: ["Recommended Shows Affinity", "Sequel", "Length Coeff", "Score Difference",
                                     "Show Popularity", "User Scored Shows", "MALAffinity",
                                     "Show Score", "Mean Score", "Standard Deviation", "User Score", "Show Name"])

    @property
    def data(self):
        if not isinstance(self._data, GeneralData):
            try:
                self._data = load_pickled_file(data_path / "general_data.pickle")
            except FileNotFoundError:
                self._data = GeneralData()
                self._data = self._data.generate_data()
                save_pickled_file(data_path / "general_data.pickle", self._data)
                self._data = self.get_means_of_OG_affs()
                save_pickled_file(data_path / "general_data.pickle", self._data)
        return self._data

    def get_means_of_OG_affs(self):
        self._data = load_pickled_file(data_path / "general_data.pickle")
        users_tag_affinity_dict = {}
        tags = Tags()
        user_db = UserDB()
        partial_main_df = user_db.df.select(user_db.stats + self._data.relevant_shows)
        try:
            OG_tag_aff_db = pd.read_parquet(data_path / "OG_tag_aff_db.parquet")
        except FileNotFoundError:
            for user_index in range(0, self._data.user_amount, 10):
                if user_index % 100 == 0:
                    print(f"Calculating affinities of each user to each tag,"
                          f" currently on user {user_index} out of {self._data.user_amount}")
                user = User(name=partial_main_df['Username'][user_index],
                            scores=partial_main_df.select(self._data.relevant_shows).row(user_index, named=True),
                            scored_amount=partial_main_df["Scored Shows"][user_index])
                user_aff_calculator = UserAffinityCalculator(user, self._data, tags)
                user = user_aff_calculator.get_user_affs()
                if not user.tag_affinity_dict:
                    continue
                users_tag_affinity_dict[user.name] = user.tag_affinity_dict

            OG_tag_aff_db = pd.DataFrame(users_tag_affinity_dict)
            OG_tag_aff_db.to_parquet(data_path / "OG_tag_aff_db.parquet")

        self._data.OG_aff_means = {}
        for j, col in enumerate(OG_tag_aff_db.index):
            self._data.OG_aff_means[col] = OG_tag_aff_db.iloc[j].mean()
        return self._data

    @property
    def major_parts(self):
        if self._major_parts is None:
            self._major_parts = self.count_major_parts()
        return self._major_parts

    @property
    def db_type(self):
        if not self._db_type:
            try:
                df = pq.ParquetFile(aff_db_path / f"{aff_db_filename}-P1.parquet")
                features = df.schema.names
                if "Max Affinity" in features:
                    self._db_type = 2
                else:
                    self._db_type = 1
            except FileNotFoundError:
                self._db_type = 1
        return self._db_type

    # @timeit
    def create(self):
        parts = int(os.cpu_count() / cpu_share)  # 1/cpu_share of available cores
        if not os.path.exists(aff_db_path / f"UserDB-P{parts}.parquet"):
            user_db = UserDB()
            user_db.split_df(parts)
            del user_db

        if not os.path.isfile(data_path / "general_data.pickle"):
            data = GeneralData()
            data.generate_data()
            save_pickled_file(data_path / "general_data.pickle", data)
            data = self.get_means_of_OG_affs()

            save_pickled_file(data_path / "general_data.pickle", data)

        print(f"Beginning creation of AffinityDB with {parts} concurrent processes")
        with ProcessPoolExecutor(max_workers=parts) as executor:
            # Each part is represented by a number from 0 to n_parts - 1
            arguments = [(i, parts) for i in range(parts)]
            _ = executor.map(self.create_minor_parts, arguments)

        print("Finished creating minor chunks. Proceeding to combine")

        self.combine()

        print("Finished combining into major parts. Proceeding to shuffle.")
        self.shuffle()

    def create_minor_parts(self,args):

        i, num_parts = args
        print(f"Entered function, process {i}")

        user_db = UserDB()
        tags = Tags()

        # # The shows in our tags dict are the ones filtered when creating it
        # # ( >= 15 min in length, >= 2 min per ep)

        # User db is divided into several parts, each one gets picked up by a separate process to work on
        df_part_i = user_db.get_df_part(i + 1)

        aff_db_dict = AffDBEntryCreator.initialize_aff_db_dict(db_type=1)
        user_amount = df_part_i.shape[0]

        # We want to save the data fairly often, otherwise the data collecting process will become slower
        minor_parts_to_create = self.total_minor_parts/num_parts

        save_data_per = user_amount // (self.total_minor_parts / num_parts)
        t1 = time.perf_counter()

        for user_index in range(0, user_amount):
            if user_index % 10 == 0:
                print(time_at_current_point(t1))
                print(f"Currently on user {user_amount * i + user_index}")

            user_scores = df_part_i.row(user_index, named=True)
            user_scored_amount = user_scores.pop("Scored Shows")
            [user_scores.pop(key, None) for key in UserDB.stats]

            user = User(name=df_part_i['Username'][user_index],
                        scores=user_scores,
                        scored_amount=user_scored_amount)

            aff_db_entry_creator = AffDBEntryCreator(user, self.data, tags)
            aff_db_entry_creator.create_db_entries_from_user_list(db_type=1, for_predict=False)

            user_db_entries = aff_db_entry_creator.aff_db_entry_dict

            for stat, stat_values in user_db_entries.items():
                # We concatenate the existing db values with the ones returned from aff_db_entry_creator
                aff_db_dict[stat] = aff_db_dict[stat] + stat_values

            if (user_index + 1) % save_data_per == 0:
                subpart_num = (user_index + 1) // save_data_per
                print(f"Finished processing user {user_amount * i + user_index + 1},"
                      f" saving PP-{int(minor_parts_to_create * i + subpart_num)}")

                for key in aff_db_dict.keys():
                    aff_db_dict[key] = np.array(aff_db_dict[key], dtype=np.float32)

                pl.DataFrame(aff_db_dict).write_parquet(data_path / "Partials" /
                    f"{aff_db_filename}-PP{int(minor_parts_to_create * i + subpart_num)}.parquet")


                # Each batch of save_data_per users will be separated into their own mini-database during
                # runtime to avoid blowing up my poor 32GB of RAM

                # After saving the data, we need to reinitialize the dicts to avoid wasting memory

                aff_db_dict = AffDBEntryCreator.initialize_aff_db_dict(db_type=1)

    @staticmethod
    def combine(max_examples_per_df=1_500_000):
        """Takes the small chunks created by create_affinity_DB and turns them into larger
        chunks, each of which is a separate database of ~1,500,000 records."""

        def count_of_minor_chunks():
            count = 0
            for dirpath, dirs, files in os.walk("Partials"):
                for filename in files:
                    if filename.startswith(f"{aff_db_filename}-PP"):
                        count += 1
            return count

        print(f"Unpacking minor chunk 1 of database")
        j = 1
        df = pl.read_parquet(aff_db_path / f"{aff_db_filename}-PP1.parquet")
        size = count_of_minor_chunks()
        pbar = tqdm(leave=True, desc="Combining", unit=" major chunks", colour="blue")
        for i in range(2, 100000):
            try:
                temp_df = pl.read_parquet(aff_db_path / f"{aff_db_filename}-PP{i}.parquet")
                df = df.vstack(temp_df)
            except FileNotFoundError:
                df.write_parquet(aff_db_path / f"{aff_db_filename}-P{j}.parquet")
                break
            if len(df) > max_examples_per_df:
                df.write_parquet(aff_db_path / f"{aff_db_filename}-P{j}.parquet")
                df = pl.DataFrame()
                pbar.total = int(np.ceil(size/i*j))
                pbar.update(1)
                j += 1
                print(f"Currently on minor chunk {i}")

    def shuffle(self):
        p = self.major_parts
        pbar = tqdm(total=p*p, leave=True, desc="Shuffling", unit=" minor chunks", colour="#CF9FFF")
        # To understand how the shuffling works, let's follow an example of the very first user in the database.
        # The example assumes the database is split into 25 major chunks, and that the user has 519 entries.

        # User 1's 519 entries are originally located in PP-1 (minor chunk 1) which has been combined into P-1
        # (major chunk 1). The issue - ALL of user 1's entries are in P-1. We want them to be spread across all
        # major chunks (P-1, P-2, ... P-25) for the database to be fully shuffled.
        for i in range(p):
            df = pd.read_parquet(aff_db_path / f"{aff_db_filename}-P{i + 1}.parquet")
            # First, we shuffle P-1 completely, making it so that user 1's 519 entries
            # are shuffled across all of P-1 (rather than just being at the start of it).

            df = shuffle_df(df)

            # Then, we split P-1 into PPS-1, PPS-2, ..., PPS-25.
            # Each of those minor semi-shuffled chunks likely has at least one entry from User 1.
            sub_size = int(len(df) / p)
            for j in range(p):
                df[j * sub_size:(j + 1) * sub_size].to_parquet(
                    aff_db_path / f"{aff_db_filename}-PPS{p * i + j + 1}.parquet")
                pbar.update(1)
                print(f"Currently on square-minor chunk {p*i + j + 1}")

        print("Finished shuffling all major parts. Beginning to recombine.")
        pbar2 = tqdm(total=p, leave=True, desc="Recombining", unit=" major chunks", colour="#CF9FFF")
        # Our goal is for User 1 to be in every major chunk. So we'll recombine in the following way :
        # PPS-1 will be part of the new P-1, PPS-2 will be part of the new P-2, and so on until
        # PPS-25 which will be part of P-25. This way, all major chunks from P-1 to P-25 have User 1's entries
        # in them. Then rinse and repeat (so P-1 will consist of PPS-1, PPS-26, PPS-51, .... etc)
        for i in range(p):
            df = pd.DataFrame()
            for j in range(i, p * p, p):
                chunk = pd.read_parquet(aff_db_path / f"{aff_db_filename}-PPS{j + 1}.parquet")
                df = pd.concat([df, chunk], ignore_index=True)
            df.to_parquet(aff_db_path / f"{aff_db_filename}-P{i + 1}.parquet")
            pbar2.update(1)

    def normalize(self):
        p = self.major_parts
        aff_means = None
        aff_stds = None
        pbar = tqdm(total=p, leave=True, desc="Normalizing", unit=" major chunks", colour="green")
        for i in range(p):
            df = pd.read_parquet(aff_db_path / f"{aff_db_filename}-P{i + 1}.parquet")
            df = self.normalize_aff_df(data=self.data, df=df, is_aff_db=True)
            df.to_parquet(aff_db_path / f"{aff_db_filename}-P{i + 1}-N.parquet")
            pbar.update(1)
            if aff_means and aff_stds:
                for col in aff_means.keys():
                    aff_means[col] = (self.data.aff_means[col] + aff_means[col] * i) / (i + 1)
                    aff_stds[col] = (self.data.aff_stds[col] + aff_stds[col] * i) / (i + 1)
            else:
                aff_means = self.data.aff_means
                aff_stds = self.data.aff_stds

        self.data.aff_means = aff_means
        self.data.aff_stds = aff_stds
        save_pickled_file(data_path / "general_data.pickle", self.data)

    @staticmethod
    def normalize_aff_df(data=None, df=None, og_filename=None, for_predict=False, is_aff_db=False):
        if og_filename:
            if os.path.isfile(f"{og_filename}-N.parquet"):
                # If a normalized version of a specific database we want to normalize already
                # exists, just load it and return that, else normalize.
                return pl.read_parquet(f"{og_filename}-N.parquet")

        if df is None:
            raise ValueError("No df and no filename were provided")

        # These two columns are normalized the same way regardless of for_predict, not with z-score.
        df["Show Popularity"] = np.log10(df["Show Popularity"] / 2) / 1.5 - 3
        df["User Scored Shows"] = (np.log10(df["User Scored Shows"]) - 2.5) * 2

        # The other three columns excluded here should not be normalized at all.
        cols_for_norm = [x for x in list(df.columns) if x not in
                         ["Pos Affinity Ratio", "Neg Affinity Ratio",
                          "Show Popularity", "User Scored Shows", "User Score",
                          "Sequel", "Score Difference", "Length Coeff"]]

        df = handle_nans(df)

        if for_predict:
            # If we're normalizing the database made from a user's list, we must use the main database's
            # mean and standard deviation, as they will be different from the mini-affinity database
            # created from the user's list.
            if not data:
                raise ValueError("Data must be sent to acquire mean of main affinity database"
                                 "if for_predict=True")
            for col in cols_for_norm:
                df.loc[df[col] != 0, col] = \
                    (df.loc[df[col] != 0, col] - data.aff_means[col]) / data.aff_stds[col]
        elif is_aff_db:
            mean_dict = {}
            std_dict = {}

            # Regular database normalization
            for col in cols_for_norm:
                mean_dict[col] = df.loc[df[col] != 0, col].mean()
                std_dict[col] = df.loc[df[col] != 0, col].std()
                if mean_dict[col] == np.nan:
                    mean_dict[col] = 0
                    std_dict[col] = 0
                df.loc[df[col] != 0, col] = (df.loc[df[col] != 0, col] - mean_dict[col]) / std_dict[col]
            data.aff_means = mean_dict
            data.aff_stds = std_dict
            save_pickled_file(data_path / "general_data.pickle", data)

        return df

    @staticmethod
    def filter_df_for_model(df):
        cols_to_take = [x for x in df.columns if not x.startswith("Doubles-")
                        and x != 'Show Score' and x != 'Show Popularity'
                        and 'Tag Count' not in x and x != 'Score Difference']  # add score difference to this
        df = df[cols_to_take]
        return df

    @staticmethod
    def remove_columns_for_model():
        p = AffinityDB.count_major_parts()
        pbar = tqdm(total=p, leave=True, desc="Removing excess columns from", unit=" major chunks", colour="#888888")
        for i in range(p):
            df = pd.read_parquet(aff_db_path / f"{aff_db_filename}-P{i + 1}-N.parquet")
            df = AffinityDB.filter_df_for_model(df)
            df.to_parquet(aff_db_path / f"{aff_db_filename}-P{i + 1}-N-{model_filename_suffix}.parquet")
            pbar.update(1)

    @staticmethod
    def count_major_parts():
        p = 1
        while True:
            if not os.path.exists(aff_db_path / f"{aff_db_filename}-P{p}.parquet"):
                break
            p += 1
        return p - 1

    def get_num_features(self):
        if os.path.exists(aff_db_path / f"{aff_db_filename}-P1-N-{model_filename_suffix}.parquet"):
            aff_db = pq.ParquetFile(aff_db_path / f"{aff_db_filename}-P1-N-{model_filename_suffix}.parquet")
        else:

            if not os.path.exists(aff_db_path / f"{aff_db_filename}-P1.parquet"):
                if not os.path.exists(aff_db_path / f"{aff_db_filename}-PP499.parquet"):
                    print("Creating Affinity DB")
                    self.create()
                else:
                    self.combine()
                    self.shuffle()

            if not os.path.exists(aff_db_path / f"{aff_db_filename}-P1-N.parquet"):
                print("Normalizing Affinity DB")
                self.normalize()
            self.remove_columns_for_model()
            aff_db = pq.ParquetFile(aff_db_path / f"{aff_db_filename}-P1-N-{model_filename_suffix}.parquet")
        num_features = len(aff_db.schema.names) - 1
        return num_features
