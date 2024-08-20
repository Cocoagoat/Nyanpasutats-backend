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
from .GeneralData import GeneralData as GeneralData
from .UserAffinityCalculator import UserAffinityCalculator
from .filenames import *
from .Graphs2 import Graphs2
from .GlobalValues import cpu_share
from concurrent.futures import ProcessPoolExecutor
import time
from dataclasses import dataclass, field
from .general_utils import load_pickled_file, save_pickled_file, time_at_current_point, shuffle_df, handle_nans

class AffDBEntryCreator:

    # Refactor this into v2 later

    def __init__(self, user: User, data: GeneralData, tags: Tags):
        self.user = user
        self.data = data
        self.features = AffinityDB.Features()
        self.tags = tags
        self.user_affinity_calculator = UserAffinityCalculator(self.user, self.data, self.tags)
        self.aff_db_entry_dict = None
        self.graphs = Graphs2()

    @classmethod
    def initialize_aff_db_dict(cls, db_type=1, for_predict=False):
        """Creates dictionary with each of the database's columns pointing to an empty list
        to which entries will be appended in the future."""

        features = AffinityDB.Features()
        tags = Tags()
        # print_attributes(tags)
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
            return 0,0

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
        entry_tags_dict = self.tags.entry_tags_dict2[entry]
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

        # if not (tag_type == 'Double' and cat_suffix.isnumeric()):
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

    def get_user_db(self, shows_to_take):
        # move to affinitydb
        # aff_db_entry_creator = AffDBEntryCreator(user, self.data, self.tags)
        self.create_db_entries_from_user_list(
            shuffle_list=False, shows_to_take=shows_to_take,
            db_type=1, for_predict=True)

        user_db_entries = self.aff_db_entry_dict
        user_shows_df_with_name = pd.DataFrame(user_db_entries)
        return user_shows_df_with_name

    def create_db_entries_from_user_list(self, shuffle_list=False, for_predict=False,
                                         shows_to_take="watched", db_type=1, sample_size=0):
        def get_user_entry_list(shows_to_take):
            if shows_to_take == "watched":
                # This will always be used for calculating affinities to create the affinity database.
                self.user.entry_list = [key for key, value in self.user.scores.items() if value]
            elif shows_to_take == "watched+related":

                temp_list = []
                self.user.entry_list = [key for key, value in self.user.scores.items() if value]
                for entry in self.user.entry_list:
                    root, related_shows = self.graphs.all_graphs_no_low_scores.find_related_entries(entry)
                    temp_list += related_shows
                self.user.entry_list = list(set(self.user.entry_list + temp_list))
            elif shows_to_take == "unwatched":
                # For predicting scores of shows the user hasn't watched yet.
                del self.aff_db_entry_dict['User Score']
                self.user.entry_list = [entry for entry in self.tags.entry_tags_dict_nls.keys()
                                        if entry in self.data.relevant_shows and not self.user.scores[entry]]
            elif shows_to_take == "both":
                # Predicting on both unwatched and watched shows (to test performance)
                del self.aff_db_entry_dict['User Score']
                self.user.entry_list = [entry for entry in self.tags.show_tags_dict_nls.keys()
                                        if entry in self.data.relevant_shows]
            elif shows_to_take == "all":
                # For predicting scores of every single entry (rather than just every show).
                del self.aff_db_entry_dict['User Score']
                self.user.entry_list = [entry for entry in self.tags.entry_tags_dict_nls.keys()
                                        if entry in self.data.relevant_shows]
            else:
                raise ValueError("shows_to_predict expects a value of either 'watched', 'unwatched', 'both' or 'all'.")

        self.aff_db_entry_dict = self.initialize_aff_db_dict(db_type, for_predict)
        #optimize above

        self.user = self.user_affinity_calculator.get_user_affs()
        get_user_entry_list(shows_to_take)
        if shuffle_list:
            random.shuffle(self.user.entry_list)
        if sample_size:
            self.user.entry_list = self.user.entry_list[0:sample_size]

        processed_entries = []
        for entry in self.user.entry_list:

            if entry in processed_entries:
                continue
            try:
                main_entry = self.tags.entry_tags_dict_nls[entry]['Main']
            except KeyError:
                continue
            main_show_data = self.tags.show_tags_dict_nls[self.tags.entry_tags_dict_nls[entry]['Main']]
            self.user.adj_tag_affinity_dict = self.user.tag_affinity_dict.copy()
            self.user.adj_pos_tag_affinity_dict = self.user.tag_pos_affinity_dict.copy()

            self.user_affinity_calculator.recalc_affinities_2(main_entry)

            for entry, length_coeff in self.tags.show_tags_dict_nls[main_entry]['Related'].items():

                # if entry in processed_entries: # try dict instead?
                #     continue
                processed_entries.append(entry)
                if shows_to_take == 'watched' and not self.user.scores[entry]:
                    continue

                if shows_to_take == "":  # can never happen, remove this later after checking
                    # In this case, the entry's tags will be the entire show's tags (so the combination
                    # of all tags for all of the entries of that show, as done in Tags.get_show_tags())
                    # This might not accurately represent a specific entry if, for example, a season 2
                    # is suddenly of a considerably different genre than the first season of the same show,
                    # but in general is better because sequels tend to have less engagement from the user base
                    # which means less people deciding on the tags on Anilist + in most cases
                    # seasons of the same show should be similar in terms of what kind of people will enjoy them
                    # so might as well make them exactly the same to avoid inconsistencies.
                    main_show_data = self.tags.show_tags_dict_nls[self.tags.entry_tags_dict_nls[entry]['Main']]
                    entry_tags_list = main_show_data['Tags'] + main_show_data['DoubleTags']
                    entry_genres_list = self.tags.show_tags_dict_nls[self.tags.entry_tags_dict_nls[entry]['Main']]['Genres']
                else:
                    # In this case, we'll take the individual tags of each entry (for testing).
                    entry_tags_list = self.tags.entry_tags_dict_nls[entry]['Tags'] + self.tags.entry_tags_dict_nls[entry]['DoubleTags']
                    entry_genres_list = self.tags.entry_tags_dict_nls[entry]['Genres']
                entry_studio = self.tags.entry_tags_dict_nls[entry]['Studio']
                entry_tag_names = [tag['name'] for tag in entry_tags_list]
                entry_tag_percentages = [tag['percentage'] for tag in entry_tags_list]

                if entry == main_entry:
                    self.aff_db_entry_dict['Sequel'].append(0)
                    self.aff_db_entry_dict['Score Difference'].append(0)
                else:
                    try:
                        self.aff_db_entry_dict['Score Difference'].append(self.data.mean_score_per_show[entry] - \
                                                                       self.data.mean_score_per_show[main_entry])
                    except (ColumnNotFoundError, KeyError) as e:
                        continue  # A unique case of a sequel that meets the conditions of partial_anime_df +
                        # a main show that doesn't (for example sequel rated 6.8 but main show rated 6.1).
                        # We don't want to count this.
                    self.aff_db_entry_dict['Sequel'].append(1)

                self.get_recommended_show_aff(entry)

                if for_predict:
                    self.aff_db_entry_dict['Show Name'].append(entry)

                if db_type == 1:
                    overall_temp_dict = self.initialize_temp_dict()
                    # processed_tags = []

                    for category, category_tags in self.tags.tags_per_category.items():
                        # entry_tags_dict = self.tags.entry_tags_dict2[entry] # move dis up
                        self.get_affinity_stats(entry, category, category_tags,
                                                overall_temp_dict)
                        # processed_tags = processed_tags + category_tags

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

                elif db_type == 2:
                    entry_tag_names = entry_tag_names + entry_genres_list + [entry_studio]

                    if self.user.scores[entry] and for_predict:
                        self.user_affinity_calculator.recalc_affinities_without_entry(entry)
                        use_adj = True
                    else:
                        use_adj = False

                    for tag_name in self.tags.all_anilist_tags:
                        aff_tag_weight, pos_aff_tag_weight = self.get_tag_weight(tag_name, entry_tag_names,
                                                                                 entry_tag_percentages,
                                                                                 use_adj = use_adj)
                        self.aff_db_entry_dict[f"{tag_name} Affinity"].append(aff_tag_weight)
                        self.aff_db_entry_dict[f"{tag_name} Pos Affinity"].append(pos_aff_tag_weight)

                if shows_to_take.startswith("watched"):
                     self.aff_db_entry_dict['User Score'].append(self.user.scores[entry])

                show_score = self.data.mean_score_per_show[entry]
                self.aff_db_entry_dict['Show Score'].append(show_score)
                self.aff_db_entry_dict['Mean Score'].append(self.user.mean_of_watched)
                self.aff_db_entry_dict['Standard Deviation'].append(self.user.std_of_watched)

                entry_members = self.data.scored_members_per_show[entry][0]
                self.aff_db_entry_dict['Show Popularity'].append(entry_members)
                self.aff_db_entry_dict['User Scored Shows'].append(self.user.scored_amount)
                self.aff_db_entry_dict['Length Coeff'].append(
                    self.tags.show_tags_dict_nls[main_entry]['Related'][entry])

        for key in list(self.aff_db_entry_dict.keys()):
            if key.startswith('Doubles'):
                del self.aff_db_entry_dict[key]

    def get_recommended_show_aff(self, entry):

        total_rec_rating = sum(
            [rating for title, rating in self.tags.entry_tags_dict_nls[entry]['Recommended'].items()])
        recommended_shows = self.tags.entry_tags_dict_nls[entry]['Recommended']
        rec_affinity = 0

        for rec_anime, rec_rating in recommended_shows.items():
            if rec_anime in self.data.relevant_shows and self.user.scores[rec_anime] and rec_rating > 0:
                MAL_score = self.data.mean_score_per_show[rec_anime]
                MAL_score_coeff = -0.6 * MAL_score + 5.9
                user_diff = self.user.scores[rec_anime] - self.user.mean_of_watched
                MAL_diff = self.data.mean_score_per_show[rec_anime] - self.user.MAL_mean_of_watched
                rec_affinity += (user_diff - MAL_diff * MAL_score_coeff) * rec_rating

        try:
            weighted_aff = rec_affinity / total_rec_rating
            if np.isnan(weighted_aff) or np.isinf(weighted_aff):
                raise ValueError
            self.aff_db_entry_dict['Recommended Shows Affinity'].append(weighted_aff)
        except (ZeroDivisionError, ValueError) as e:  # No relevant recommended shows
            self.aff_db_entry_dict['Recommended Shows Affinity'].append(0)


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
        # All properties are loaded on demand
        self._df = None
        self._data = None
        self.features = self.Features()
        self._OG_aff_means = None
        self._major_parts = None
        self._db_type = db_type
        self.graphs = Graphs2()

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
                                     "Show Popularity", "User Scored Shows",
                                     "Show Score", "Mean Score", "Standard Deviation", "User Score", "Show Name"])

    @property
    def data(self):
        if not isinstance(self._data, GeneralData):
            try:
                self._data = load_pickled_file(data_path / "general_data.pickle")
            except FileNotFoundError:
                print("GeneralData not found, proceeding to load")
                self._data = GeneralData()
                self._data = self._data.generate_data()
                self._data = self.get_means_of_OG_affs()
                # self._get_general_data()
                save_pickled_file(data_path / "general_data.pickle", self._data)
        return self._data

    def get_means_of_OG_affs(self):
        self._data = load_pickled_file(data_path / "general_data.pickle")
        users_tag_affinity_dict = {}
        tags = Tags()
        user_db = UserDB()
        partial_main_df = user_db.df.select(user_db.stats + self._data.relevant_shows)
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
            # time.sleep(5)

        if not os.path.isfile(data_path / "general_data.pickle"):
            data = GeneralData()
            data.generate_data()
            save_pickled_file(data_path / "general_data.pickle", data)
            data = self.get_means_of_OG_affs()

            save_pickled_file(data_path / "general_data.pickle", data)

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
        # anime_db = AnimeDB()
        tags = Tags()

        # # The shows in our tags dict are the ones filtered when creating it
        # # ( >= 15 min in length, >= 2 min per ep)

        # User db is divided into several parts, each one gets picked up by a separate process to work on
        df_part_i = user_db.get_df_part(i + 1)

        partial_main_df = df_part_i.select(user_db.stats + self.data.relevant_shows)
        aff_db_dict = AffDBEntryCreator.initialize_aff_db_dict(db_type=1)
        user_amount = partial_main_df.shape[0]

        # We want to save the data fairly often, otherwise the data collecting process will become slower
        minor_parts_to_create = self.total_minor_parts/num_parts

        save_data_per = user_amount // (self.total_minor_parts / num_parts)
        t1 = time.perf_counter()

        # get_means_of_OG_affs()
        # Add OG_data thingy here and not in GeneralData?
        for user_index in range(user_amount):
            if user_index % 10 == 0:
                print(time_at_current_point(t1))
                print(f"Currently on user {user_amount * i + user_index}")

            user = User(name=partial_main_df['Username'][user_index],
                        scores=partial_main_df.select(self.data.relevant_shows).row(user_index, named=True),
                        scored_amount=partial_main_df["Scored Shows"][user_index])

            user = UserAffinityCalculator.initialize_user_stats(user, self.data)

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
        for i in range(226, 100000):
            try:
                temp_df = pl.read_parquet(aff_db_path / f"{aff_db_filename}-PP{i}.parquet")
                df = df.vstack(temp_df)
            except FileNotFoundError:
                df.write_parquet(aff_db_path / f"{aff_db_filename}-P{j}.parquet")
                break
            if len(df) > max_examples_per_df:
                # print(f"Reached {max_examples_per_df} examples. Saving part {j}")
                df.write_parquet(aff_db_path / f"{aff_db_filename}-P{j}.parquet")
                df = pl.DataFrame()
                pbar.total = int(np.ceil(size/i*j))
                pbar.update(1)
                j += 1
                print(f"Currently on minor chunk {i}")

    def shuffle(self):
        p = self.major_parts
        pbar = tqdm(total=p*p, leave=True, desc="Shuffling", unit=" minor chunks", colour="#CF9FFF")
        # First we shuffle each major database chunk. This isn't enough for a thorough
        # shuffle - since each major chunk is currently simply a combination of

        # Then we separate each major chunk into major_part_amount smaller chunks.
        for i in range(p):
            df = pd.read_parquet(aff_db_path / f"{aff_db_filename}-P{i + 1}.parquet")
            df = shuffle_df(df)
            sub_size = int(len(df) / p)
            for j in range(p):
                df[j * sub_size:(j + 1) * sub_size].to_parquet(
                    aff_db_path / f"{aff_db_filename}-PPS{p * i + j + 1}.parquet")
                pbar.update(1)
                print(f"Currently on square-minor chunk {p*i + j + 1}")

        print("Finished shuffling all major parts. Beginning to recombine.")
        pbar2 = tqdm(total=p, leave=True, desc="Recombining", unit=" major chunks", colour="#CF9FFF")
        for i in range(p):
            df = pd.DataFrame()
            for j in range(i, p * p, p):
                # print(f"--Loading minor shuffled part {j + 1} of database--")
                chunk = pd.read_parquet(aff_db_path / f"{aff_db_filename}-PPS{j + 1}.parquet")
                df = pd.concat([df, chunk], ignore_index=True)

            # print(f"---------Writing major shuffled part {i + 1} of database----------")
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
                print("Creating Affinity DB")
                self.create()
            if not os.path.exists(aff_db_path / f"{aff_db_filename}-P1-N.parquet"):
                print("Normalizing Affinity DB")
                self.normalize()
            self.remove_columns_for_model()
            aff_db = pq.ParquetFile(aff_db_path / f"{aff_db_filename}-P1-N-{model_filename_suffix}.parquet")
        num_features = len(aff_db.schema.names) - 1
        return num_features
