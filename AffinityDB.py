import os

import general_utils
from AnimeDB import AnimeDB
from UserDB import UserDB
from Tags import Tags
from filenames import *
from MAL_utils import *
from Graphs import Graphs
from concurrent.futures import ProcessPoolExecutor
from memory_profiler import profile
import shutil
from collections import namedtuple
import time
from dataclasses import dataclass, field
from functools import singledispatch


@dataclass
class GeneralData:
    relevant_shows: list = field(default=None)

    mean_score_per_show: pl.DataFrame = field(default=None) # change those three to dicts?
    scored_members_per_show: pl.DataFrame = field(default=None)
    scored_shows_per_user: pl.Series = field(default=None)

    user_amount: int = field(default=None)

    aff_means: dict = field(default=None)
    aff_stds: dict = field(default=None)

    OG_aff_means: dict = field(default=None)
    OG_aff_stds: dict = field(default=None)

    db_type: int = field(default=None)

    # _users_tag_affinity_dict : dict = field(default=None)
    # users_tag_pos_affinity_dict : dict = field(default=None)

    def generate_data(self):
        tags = Tags()
        anime_db = AnimeDB()
        user_db = UserDB()
        self.relevant_shows = list(tags.entry_tags_dict.keys())
        partial_main_df = user_db.df.select(user_db.stats + self.relevant_shows)
        partial_anime_df = anime_db.df.select(["Rows"] + self.relevant_shows)

        self.mean_score_per_show = partial_anime_df.filter(pl.col('Rows') == "Mean Score")
        self.scored_members_per_show = partial_anime_df.filter(pl.col('Rows') == "Scores").to_dict(
            as_series=False)
        self.scored_shows_per_user = user_db.df['Scored Shows'].to_list()
        self.user_amount = partial_main_df.shape[0]
        if os.path.isfile(aff_db_filename):
            self._get_means_of_affs()
            self._get_db_type()
        self._get_means_of_OG_affs(partial_main_df)

        return self
        # self._get_means_of_OG_aff_columns()


    def _get_db_type(self):
        pass

    def _get_means_of_affs(self):
        aff_db = AffinityDB()
        mean_dict = {}
        std_dict = {}

        if aff_db.df is not None: # Think about how to include those means later? Also why would aff_db.df be equal none
            cols_for_norm = [x for x in list(aff_db.df.columns) if x not in
                             ["Pos Affinity Ratio", "Neg Affinity Ratio", "Show Popularity", "User Scored Shows",
                              "User Score"]]
            for col in cols_for_norm:
                # mean_dict[col] = aff_db.df.filter(aff_db.df[col] != 0)[col].mean()
                # std_dict[col] = aff_db.df.filter(aff_db.df[col] != 0)[col].std()
                mean_dict[col] = aff_db.df[col].mean()
                std_dict[col] = aff_db.df[col].std()
            self.aff_means = mean_dict
            self.aff_stds = std_dict

    def _get_means_of_OG_affs(self, partial_main_df):
        # user_db = UserDB()
        # anime_db = AnimeDB()
        tags = Tags()
        # partial_main_df = user_db.select(user_db.stats + self.relevant_shows)
        # partial_anime_df = anime_db.df.select(["Rows"] + self.relevant_shows)
        users_tag_affinity_dict = {}
        for user_index in range(0, self.user_amount, 10):
            if user_index%100 == 0:
                print(f"Calculating affinities of each user to each tag,"
                       f" currently on user {user_index} out of {self.user_amount}")
            user = User(name=partial_main_df['Username'][user_index],
                        scores=partial_main_df.select(self.relevant_shows).row(user_index, named=True),
                        scored_amount=partial_main_df["Scored Shows"][user_index])
            user_aff_calculator = UserAffinityCalculator(user, self, tags)
            user = user_aff_calculator.get_user_affs()
            if not user.tag_affinity_dict:
                continue
            users_tag_affinity_dict[user.name] = user.tag_affinity_dict

        OG_tag_aff_db = pd.DataFrame(users_tag_affinity_dict)
        OG_tag_aff_db.to_parquet("OG_tag_aff_db.parquet")

        self.OG_aff_means = {}
        for i,col in enumerate(OG_tag_aff_db.index):
            self.OG_aff_means[col] = OG_tag_aff_db.iloc[i].mean()


        # OG_aff_db = pl.DataFrame(self.user_tag_affinity_dict)
        # mean_dict = {}
        # std_dict = {}
        #
        # for col in OG_aff_db.columns:
        #     mean_dict[col] = OG_aff_db.filter(OG_aff_db[col] != 0)[col].mean()
        #     std_dict[col] = OG_aff_db.filter(OG_aff_db[col] != 0)[col].std()
        # self.OG_aff_db_means = mean_dict
        # self.OG_aff_db_stds = std_dict

    # def _get_means_of_OG_aff_columns(self):
    #     aff_db = AffinityDB()
    #     mean_dict = {}
    #     std_dict = {}
    #     cols_for_norm = [x for x in list(aff_db.df.columns) if x not in
    #                      ["Pos Affinity Ratio", "Neg Affinity Ratio", "Show Popularity", "User Scored Shows",
    #                       "User Score"]]
    #     for col in cols_for_norm:
    #         mean_dict[col] = aff_db.df.filter(aff_db.df[col] != 0)[col].mean()
    #         std_dict[col] = aff_db.df.filter(aff_db.df[col] != 0)[col].std()
    #     self.aff_db_means = mean_dict
    #     self.aff_db_stds = std_dict

    # @property
    # def users_tag_affinity_dict(self):
    #     if not self._users_tag_affinity_dict:
    #         try:
    #             self._users_tag_affinity_dict = load_pickled_file(user_tag_affinity_dict_filename)
    #         except FileNotFoundError:
    #             print("user_tag_affinity_dict not found, proceeding to load. This will take a while.")
    #             self.get_users_tag_affinity_dict()
    #             save_pickled_file(user_tag_affinity_dict_filename, self._users_tag_affinity_dict)
    #     return self._users_tag_affinity_dict

    # def get_users_tag_affinity_dict(self):
    #     user_db = UserDB()
    #     anime_db = AnimeDB()
    #     tags = Tags()
    #
    #     partial_main_df = user_db.df.select(user_db.stats + self.relevant_shows)
    #
    #     for user_index in range(1000):
    #         if user_index % 100 == 0:
    #             print(f"Calculating affinities, currently on user {user_index}")
    #
    #         user = User(name=partial_main_df['Username'][user_index],
    #                     scores=partial_main_df.select(self.relevant_shows).row(user_index, named=True),
    #                     scored_amount=partial_main_df["Scored Shows"][user_index])
    #         user = UserAffinityCalculator(user, self, tags).get_user_affs()
    #
    #         if not self._users_tag_affinity_dict:
    #             self._users_tag_affinity_dict = {}
    #             self.users_tag_pos_affinity_dict={}
    #             for tag in user.tag_affinity_dict:
    #                 self._users_tag_affinity_dict[tag] = []
    #                 self.users_tag_pos_affinity_dict[tag] = []
    #                 #deal with pos dict, make mean 0.5?
    #
    #         for tag in user.tag_affinity_dict:
    #             self._users_tag_affinity_dict[tag].append(user.tag_affinity_dict[tag])
    #             self.users_tag_pos_affinity_dict[tag].append(user.tag_pos_affinity_dict[tag])
    #
    #     for key in self._users_tag_affinity_dict.keys():
    #         self._users_tag_affinity_dict[key] = np.array(self._users_tag_affinity_dict[key], dtype=np.float32)
    #         self.users_tag_pos_affinity_dict[key] = np.array(self.users_tag_pos_affinity_dict[key], dtype=np.float32)


@dataclass
class User:
    tag_affinity_dict: dict = field(default=None)
    tag_pos_affinity_dict: dict = field(default=None)
    adj_tag_affinity_dict : dict = field(default=None)
    adj_pos_tag_affinity_dict : dict = field(default=None)

    mean_of_single_affs: int = field(default=None)
    mean_of_double_affs: int = field(default=None)

    name: str = field(default=None)
    scored_amount: int = field(default=None)
    scores: dict = field(default=None)

    mean_of_watched: float = field(default=None)
    std_of_watched: float = field(default=None)
    MAL_mean_of_watched: float = field(default=None)

    score_per_tag: dict = field(default=None)
    MAL_score_per_tag: dict = field(default=None)
    positive_aff_per_tag: dict = field(default=None)
    freq_coeff_per_tag: dict = field(default=None)
    tag_entry_list : dict = field(default=None)
    tag_counts: dict = field(default=None)

    sum_of_freq_coeffs_st: int = field(default=0)
    sum_of_freq_coeffs_dt: int = field(default=0)

    freq_multi_st : int = field(default=0)
    freq_multi_dt : int = field(default=0)

    entry_list: list = field(default=None)
    show_count: int = field(default=None)

    score_diffs: list = field(default=None)
    exp_coeffs: list = field(default=None)

    def center_tag_aff_dict_means(self, means):
        for tag in self.tag_affinity_dict:
            # Normalizing the affinities to each tag, with the means being calculated in generate_data().
            try:
                self.tag_affinity_dict[tag] =\
                    self.tag_affinity_dict[tag] - means[tag]
            except TypeError:
                continue

    # def adjust_aff_dicts(self, entry_tags_dict):


def initialize_user_stats(user, data):
    """Initializes the necessary stats for user's affinities to be calculated."""
    watched_user_score_list = [score for title, score in user.scores.items() if score]
    # ^ Simply a list of the user's scores for every show they watched

    watched_titles = [title for title, score in user.scores.items() if score]
    # watched_MAL_score_dict = self.data.mean_score_per_show.select(watched_titles).to_dict(as_series=False)
    watched_MAL_score_dict = data.mean_score_per_show.select(watched_titles).to_dict(as_series=False)
    watched_MAL_score_list = [score[0] for title, score in
                              watched_MAL_score_dict.items() if title != "Rows"]

    user.MAL_mean_of_watched = np.mean(watched_MAL_score_list)
    user.mean_of_watched = np.mean(watched_user_score_list)
    user.std_of_watched = np.std(watched_user_score_list)
    user.entry_list = watched_titles
    return user


#  ------------------------------------------------------------------------------------------------
class AffDBEntryCreator:

    def __init__(self, user : User, data : GeneralData, tags: Tags):
        self.user = user
        self.data = data
        self.features = AffinityDB.Features()
        self.tags = tags
        self.user_affinity_calculator = UserAffinityCalculator(self.user, self.data, self.tags)
        self.aff_db_entry_dict = None

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
                                                         if "Double" not in stat}
            aff_db_entry_dict = aff_db_entry_dict | {"Studio Affinity": [], "Studio Pos Affinity": []}

        elif db_type == 2:
            aff_db_entry_dict = {f"{tag} Affinity": [] for tag in tags.all_anilist_tags}
            aff_db_entry_dict = aff_db_entry_dict | {f"{tag} Pos Affinity": [] for tag in tags.all_anilist_tags}

        aff_db_entry_dict = aff_db_entry_dict | {stat: [] for stat in features.other_features}

        if not for_predict:
            del aff_db_entry_dict['Show Name']
        # print_attributes(tags)
        return aff_db_entry_dict

    def get_tag_weight(self, tag_name, entry_tags_list, entry_tag_percentages, use_adj=False):

        try:
            tag_index = entry_tags_list.index(tag_name)
        except ValueError:
            return 0,0

        # try:
        #     adjusted_p = self.tags.adjust_tag_percentage(entry_tag_percentages[tag_index])
        # except (KeyError, TypeError, IndexError) as e:
        #     adjusted_p = 1  # Genres and studios don't have a percentage

        try:
            adjusted_p = entry_tag_percentages[tag_index]
        except IndexError:
            adjusted_p = 1

        if adjusted_p == 0:
            return 0,0

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

    def get_affinity_stats(self, entry, category, category_tag_list, entry_tags_dict, entry_genres_list, #entry_tag_list, entry_tag_percentages
                             overall_temp_dict):
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


        # temp_dict = {"Max Affinity": None, "Min Affinity": None, "Avg Affinity": 0,
        #              "Max Pos Affinity": None, "Avg Pos Affinity": 0, "Pos Affinity Ratio": 0,
        #              "Neg Affinity Ratio": 0}  # Basic stats with no distinction between tag types
        #
        # temp_dict = {f"{tag_type} {stat}": stat_value for tag_type in Tags().tag_types

         # Give this tag_type if it's only related to overall_dict

        try:
            tag_name = category_tag_list[0]
        except IndexError:
            tag_name = None

        tag_type = "Single" if tag_name and "<" not in tag_name else "Double" # All tags in a category = same type
        cat_suffix = category.split("-")[-1]
        # There won't be any doubles with <Cute Girls Doing Cute Things>x<?> if the show isn't a CGDCT
        # if (tag_type == 'Double'
        #         and not cat_suffix.isnumeric()  # To avoid skipping Doubles-1, Doubles-2 etc
        #         and cat_suffix not in (list(entry_tags_dict.keys()) + entry_genres_list)):
        #     return

        temp_dict = self.initialize_temp_dict()
        if category == 'Genres' or category == 'Studio':
            for tag_name in category_tag_list:
                entry_tags_dict[tag_name] = {}
                entry_tags_dict[tag_name]['percentage'] = 1

        self.user_affinity_calculator.recalc_affinities_2(entry_tags_dict, entry)

        for tag_name in category_tag_list:
            # if category == 'Genres' or category == 'Studio':
            #     entry_tags_dict[tag_name] = {}
            #     entry_tags_dict[tag_name]['percentage'] = 1
            # aff_tag_weight, pos_aff_tag_weight = self.get_tag_weight(tag_name, entry_tags_list, entry_tag_percentages)
            try:
                tag_p = entry_tags_dict[tag_name]['percentage']
                # self.user.adjust_aff_dicts(entry_tags_dict, entry) # 'Crime' : 0.94 etc
                aff_tag_weight = self.user.adj_tag_affinity_dict[tag_name] * tag_p
                pos_aff_tag_weight = self.user.adj_pos_tag_affinity_dict[tag_name] * tag_p
                # tag_count += entry_tags_dict[tag_name]['percentage']
            except KeyError:
                continue
            if not aff_tag_weight:
                continue
            update_stats(temp_dict, entry_tags_dict[tag_name]['percentage'],
                         aff_tag_weight, pos_aff_tag_weight, tag_type)

        # single_tag_count = temp_dict["Single Pos Affinity Ratio"] + temp_dict["Single Neg Affinity Ratio"]
        # double_tag_count = temp_dict["Double Pos Affinity Ratio"] + temp_dict["Double Neg Affinity Ratio"]
        tag_count = temp_dict[f"{tag_type} Pos Affinity Ratio"] + temp_dict[f"{tag_type} Neg Affinity Ratio"]

        try:
            temp_dict[f"{tag_type} Avg Affinity"] /= tag_count
            temp_dict[f"{tag_type} Avg Pos Affinity"] /= tag_count
        except ZeroDivisionError:
            temp_dict[f"{tag_type} Avg Affinity"] = 0
            temp_dict[f"{tag_type} Avg Pos Affinity"] = 0

        # try:
        #     temp_dict["Double Avg Affinity"] /= double_tag_count
        #     temp_dict["Double Avg Pos Affinity"] /= double_tag_count
        # except ZeroDivisionError:
        #     temp_dict["Double Avg Affinity"] = 0
        #     temp_dict["Double Avg Pos Affinity"] = 0

        if category != 'Studio':
            for stat in self.features.category_features: #max, min, avg, maxpos
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

        if not (tag_type == 'Double' and cat_suffix.isnumeric()):
            for key in temp_dict.keys():
                if not temp_dict[key]:
                    continue
                stat_name = " ".join(key.split()[1:])
                if stat_name.startswith("Avg"):
                    # If key doesn't start with min or max, it's a counter (avg or neg/pos)
                    overall_temp_dict[key] += temp_dict[key]*tag_count #Make the single/double distinction here?
                elif stat_name.startswith("Min"):
                    if overall_temp_dict[key] is None or temp_dict[key] < overall_temp_dict[key]:
                        overall_temp_dict[key] = temp_dict[key]
                elif stat_name.startswith("Max"): # It's one of the "max" values
                    if overall_temp_dict[key] is None or temp_dict[key] > overall_temp_dict[key]:
                        overall_temp_dict[key] = temp_dict[key]
                else: # It's the pos/neg tag ratios
                    overall_temp_dict[key] += temp_dict[key]

    # def create_db_entries_from_user_list2(self, shuffle_list = True, shows_to_take="watched"):
    #     pass

    @staticmethod
    def initialize_temp_dict():
        temp_dict = {"Max Affinity": None, "Min Affinity": None, "Avg Affinity": 0,
                     "Max Pos Affinity": None, "Avg Pos Affinity": 0, "Pos Affinity Ratio": 0,
                     "Neg Affinity Ratio": 0}  # Basic stats with no distinction between tag types

        temp_dict = {f"{tag_type} {stat}": stat_value for tag_type in Tags().tag_types
                     for stat, stat_value in temp_dict.items()}  # Quick way to create stats for each tag type
        return temp_dict

    def create_db_entries_from_user_list(self, shuffle_list=True, for_predict=False, shows_to_take="watched", db_type=1,
                                         sample_size=0):
        def get_user_entry_list(shows_to_take):
            if shows_to_take == "watched":
                # This will always be used for calculating affinities to create the affinity database.
                self.user.entry_list = [key for key, value in self.user.scores.items() if value]
            elif shows_to_take == "unwatched":
                # For predicting scores of shows the user hasn't watched yet.
                del self.aff_db_entry_dict['User Score']
                self.user.entry_list = [entry for entry in self.tags.entry_tags_dict.keys()
                                        if entry in self.data.relevant_shows and not self.user.scores[entry]]
            elif shows_to_take == "both":
                # Predicting on both unwatched and watched shows (to test performance)
                del self.aff_db_entry_dict['User Score']
                self.user.entry_list = [entry for entry in self.tags.show_tags_dict.keys()
                                        if entry in self.data.relevant_shows]
            elif shows_to_take == "all":
                # For predicting scores of every single entry (rather than just every show).
                del self.aff_db_entry_dict['User Score']
                self.user.entry_list = [entry for entry in self.tags.entry_tags_dict.keys()
                                        if entry in self.data.relevant_shows]
            else:
                raise ValueError("shows_to_predict expects a value of either 'watched', 'unwatched', 'both' or 'all'.")

        self.aff_db_entry_dict = self.initialize_aff_db_dict(db_type, for_predict)

        self.user = self.user_affinity_calculator.get_user_affs()
        get_user_entry_list(shows_to_take)
        if shuffle_list:
            random.shuffle(self.user.entry_list)
        if sample_size:
            self.user.entry_list = self.user.entry_list[0:sample_size]

        # processed_main_entries = []
        for entry in self.user.entry_list:
            main_entry = self.tags.entry_tags_dict[entry]['Main']
            for related_entry, length_coeff in self.tags.show_tags_dict[main_entry]['Related'].items():
                pass
            # We only want 1 entry from each show for each user in our database
            # (To avoid situations such as having data from how much 1 user likes 45 Detective Conan movies)
            # if not for_predict and main_entry in processed_main_entries:
            #     continue
            # processed_main_entries.append(main_entry)

            if shows_to_take == "pain": #basically always, fix this later
                # In this case, the entry's tags will be the entire show's tags (so the combination
                # of all tags for all of the entries of that show, as done in Tags.get_show_tags())
                # This might not accurately represent a specific entry if, for example, a season 2
                # is suddenly of a considerably different genre than the first season of the same show,
                # but in general is better because sequels tend to have less engagement from the user base
                # which means less people deciding on the tags on Anilist + in most cases
                # seasons of the same show should be similar in terms of what kind of people will enjoy them
                # so might as well make them exactly the same to avoid inconsistencies.
                main_show_data = self.tags.show_tags_dict[self.tags.entry_tags_dict[entry]['Main']]
                entry_tags_list = main_show_data['Tags'] + main_show_data['DoubleTags']
                entry_genres_list = self.tags.show_tags_dict[self.tags.entry_tags_dict[entry]['Main']]['Genres']
            else:
                # In this case, we'll take the individual tags of each entry (for testing).
                entry_tags_list = self.tags.entry_tags_dict[entry]['Tags'] + self.tags.entry_tags_dict[entry]['DoubleTags']
                entry_genres_list = self.tags.entry_tags_dict[entry]['Genres']
            entry_studio = self.tags.entry_tags_dict[entry]['Studio']
            entry_tag_names = [tag['name'] for tag in entry_tags_list]
            entry_tag_percentages = [tag['percentage'] for tag in entry_tags_list]

            # Testing
            entry_tags_dict = {}
            for tag in entry_tags_list:
                if "<" in tag['name']:
                    entry_tags_dict[tag['name']] = {'percentage': tag['percentage']}
                else:
                    entry_tags_dict[tag['name']] = {'percentage': tag['percentage'], 'category': tag['category']}

            if entry == main_entry:
                self.aff_db_entry_dict['Sequel'].append(0)
                self.aff_db_entry_dict['Score Difference'].append(0)
            else:
                try:
                    self.aff_db_entry_dict['Score Difference'].append(self.data.mean_score_per_show[entry].item() - \
                                                                   self.data.mean_score_per_show[main_entry].item())
                except ColumnNotFoundError:
                    continue # A unique case of a sequel that meets the conditions of partial_anime_df +
                    # a main show that doesn't. We don't want to count this.
                self.aff_db_entry_dict['Sequel'].append(1)

            self.get_recommended_show_aff(entry)

            if for_predict:
                self.aff_db_entry_dict['Show Name'].append(entry)

            if db_type == 1:
                overall_temp_dict = self.initialize_temp_dict()
                processed_tags = []

                for category, category_tags in self.tags.tags_per_category.items():

                    # if category == 'Doubles-1': # Doubles-1 and Doubles-30
                    #     break
                    #     try:
                    #         cat_suffix = int(cat_suffix) # We
                    #         continue
                    #     except ValueError:
                    #         for tag in entry_tag_names:
                    #             if tag == cat_suffix:
                    #                 break
                    #             if
                    self.get_affinity_stats(entry, category, category_tags,
                                            entry_tags_dict,entry_genres_list, overall_temp_dict)
                    processed_tags = processed_tags + category_tags

                # category = "Remaining"
                # remaining_tags = [tag for tag in entry_tag_names if tag not in processed_tags]
                # self.get_affinity_stats(category, remaining_tags, entry_tags_dict,
                #                         entry_genres_list, overall_temp_dict)

                category = "Genres"
                self.get_affinity_stats(entry, category, entry_genres_list, {}, [], overall_temp_dict)

                category = "Studio"
                self.get_affinity_stats(entry, category, [entry_studio], {}, [], overall_temp_dict)

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

            if self.user.scores[entry] and shows_to_take == "watched":
                # tf is this?
                self.aff_db_entry_dict['User Score'].append(self.user.scores[entry])

            # self.aff_db_entry_dict['Show Score'].append(self.data.mean_score_per_show[entry][0])
            # show_score = self.tags.get_avg_score_of_show(entry, self.data.mean_score_per_show)
            show_score = self.data.mean_score_per_show[entry].item()
            self.aff_db_entry_dict['Show Score'].append(show_score)
            self.aff_db_entry_dict['Mean Score'].append(self.user.mean_of_watched)
            self.aff_db_entry_dict['Standard Deviation'].append(self.user.std_of_watched)

            entry_members = self.data.scored_members_per_show[entry][0]
            self.aff_db_entry_dict['Show Popularity'].append(entry_members)

            self.aff_db_entry_dict['User Scored Shows'].append(self.user.scored_amount)

            # main_entry = self.tags.entry_tags_dict[entry]['Main']
            # if entry == main_entry:
            #     self.aff_db_entry_dict['Sequel'].append(0)
            #     self.aff_db_entry_dict['Score Difference'].append(0)
            # else:
            #     self.aff_db_entry_dict['Sequel'].append(1)
            #     try:
            #         self.aff_db_entry_dict['Score Difference'].append(self.data.mean_score_per_show[entry].item() - \
            #                                                       self.data.mean_score_per_show[main_entry].item())
            #     except ColumnNotFoundError:
            #         print(5)

            self.aff_db_entry_dict['Length Coeff'].append(self.tags.show_tags_dict[main_entry]['Related'][entry])

    def get_recommended_show_aff(self, entry):

        total_rec_rating = sum(
            [rating for title, rating in self.tags.entry_tags_dict[entry]['Recommended'].items()])
        recommended_shows = self.tags.entry_tags_dict[entry]['Recommended']
        rec_affinity = 0

        for rec_anime, rec_rating in recommended_shows.items():
            if rec_anime in self.data.relevant_shows and self.user.scores[rec_anime] and rec_rating > 0:
                MAL_score = self.data.mean_score_per_show[rec_anime][0]
                MAL_score_coeff = -0.6 * MAL_score + 5.9
                user_diff = self.user.scores[rec_anime] - self.user.mean_of_watched
                MAL_diff = self.data.mean_score_per_show[rec_anime][0] - self.user.MAL_mean_of_watched
                rec_affinity += (user_diff - MAL_diff * MAL_score_coeff) * rec_rating

        try:
            weighted_aff = rec_affinity / total_rec_rating
            if np.isnan(weighted_aff) or np.isinf(weighted_aff):
                raise ValueError
            self.aff_db_entry_dict['Recommended Shows Affinity'].append(weighted_aff)
        except (ZeroDivisionError, ValueError) as e:  # No relevant recommended shows
            self.aff_db_entry_dict['Recommended Shows Affinity'].append(0)

    # def get_categories_aff(self):
    #     overall_temp_dict = {"Max Affinity": None, "Min Affinity": None, "Avg Affinity": 0,
    #                          "Max Pos Affinity": None, "Avg Pos Affinity": 0, "Pos Affinity Ratio": 0,
    #                          "Neg Affinity Ratio": 0}
    #     processed_tags = []
    #     entry_tag_names = [tag['name'] for tag in self.entry_tags_list]
    #     entry_tag_percentages = [tag['percentage'] for tag in self.entry_tags_list]


#  ----------------------------------------------------------------------------------------------------
class UserAffinityCalculator:

    def __init__(self, user: User, data: GeneralData, tags: Tags):
        self.user = user
        self.data = data
        self.tags = tags

    def initialize_user_dicts(self):
        self.user.tag_affinity_dict = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.tag_pos_affinity_dict = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.score_per_tag = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.MAL_score_per_tag = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.tag_entry_list = {tag_name: [] for tag_name in self.tags.all_anilist_tags}
        self.user.freq_coeff_per_tag = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        # self.user.positive_aff_per_tag = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.tag_counts = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}

    def _calculate_affinities(self, tag, recalc=False, show_to_exclude=None,
                              entry_pos_aff=None, sum_of_length_coeffs=None):
        if recalc:
            try:
                p = self.tags.adjust_tag_percentage(tag['percentage'])
                if p == 0:
                    return
                tag = tag['name']
            except TypeError:
                p = 1

            # sum_of_length_coeffs = sum([length_coeff for entry, length_coeff
            #                             in self.tags.show_tags_dict[show_to_exclude]['Related'].items()
            #                             if self.user.scores[entry]])

            entries_to_exclude = [entry for entry in self.tags.show_tags_dict[show_to_exclude]['Related'].keys()]
            score_of_excluded_show = sum([self.user.scores[entry]*length_coeff/sum_of_length_coeffs
                                          for entry, length_coeff in
                                          self.tags.show_tags_dict[show_to_exclude]['Related'].items()
                                          if self.user.scores[entry]])

            try:
                tag_overall_ratio = (self.user.tag_counts[tag]-1)/(self.user.show_count-1)
                freq_coeff = min(1, max((self.user.tag_counts[tag] - 1) / 10, tag_overall_ratio * 20))

                # score_of_excluded_show = self.user.scores[show_to_exclude]
                user_tag_diff = (self.user.score_per_tag[tag] - score_of_excluded_show * p) / \
                                (self.user.tag_counts[tag] - p) - self.user.mean_of_watched

                MAL_score_of_excluded_show = self.data.mean_score_per_show[show_to_exclude].item()
                MAL_tag_diff = (self.user.MAL_score_per_tag[tag] - MAL_score_of_excluded_show * p) / \
                               (self.user.tag_counts[tag] - p) - self.user.MAL_mean_of_watched

                self.user.adj_tag_affinity_dict[tag] = (2 * user_tag_diff - MAL_tag_diff) * freq_coeff
                try:

                    # If anyone ever sees this, I'm sorry, it's 3AM here, my brain gave up, send help
                    # (it's basically the pos aff of one show, accounting for all the entries the user watched)
                    # entry_pos_aff = sum([self.user.exp_coeffs[10-self.user.scores[entry]]*
                    #                        self.MAL_score_coeff(self.data.mean_score_per_show[entry].item())*
                    #                        length_coeff/sum_of_length_coeffs
                    #                        for entry, length_coeff in self.tags.show_tags_dict[show_to_exclude]['Related'].items()
                    #                        if self.user.scores[entry] and
                    #                        self.user.scores[entry] > self.user.mean_of_watched])
                    entry_pos_aff /= self.user.tag_counts[tag]
                    entry_pos_aff *= self.user.freq_coeff_per_tag[tag]
                except IndexError:
                    entry_pos_aff = 0
                except ZeroDivisionError:
                    entry_pos_aff = self.user.adj_tag_affinity_dict[tag] # set pos aff to 0 if no shows with tag left

# is freq coeff even included in the OG pos aff calculation?

                self.user.adj_pos_tag_affinity_dict[tag] -= entry_pos_aff

                self.user.adj_pos_tag_affinity_dict[tag] *= (freq_coeff/self.user.freq_coeff_per_tag[tag])

                # Normalize

                self.user.adj_tag_affinity_dict[tag] = \
                    self.user.adj_tag_affinity_dict[tag] - self.data.OG_aff_means[tag] -\
                    self.user.mean_of_affs * self.freq_multi * self.user.freq_coeff_per_tag[tag]

                # self.user.adj_pos_tag_affinity_dict[tag] = \
                #     self.user.adj_pos_tag_affinity_dict[tag] - self.data.OG_aff_means[tag] - \
                #     self.user.mean_of_affs * self.freq_multi * self.user.freq_coeff_per_tag[tag]

            except ZeroDivisionError:
                self.user.adj_tag_affinity_dict[tag] = 0
                self.user.adj_pos_tag_affinity_dict[tag] = 0
        else:
            try:
                tag_overall_ratio = self.user.tag_counts[tag] / self.user.show_count
                freq_coeff = min(1, max(self.user.tag_counts[tag] / 7, tag_overall_ratio * 20))
                if "<" in tag:
                    self.user.sum_of_freq_coeffs_dt += freq_coeff
                else:
                    self.user.sum_of_freq_coeffs_st += freq_coeff
                self.user.freq_coeff_per_tag[tag] = freq_coeff
                # User has to watch either at least 7 shows with the tag or have the tag
                # in at least 5% of their watched shows for it to count fully.
                user_tag_diff = self.user.score_per_tag[tag] / self.user.tag_counts[tag] - self.user.mean_of_watched
                MAL_tag_diff = self.user.MAL_score_per_tag[tag] / self.user.tag_counts[tag] - self.user.MAL_mean_of_watched
                self.user.tag_affinity_dict[tag] = (2 * user_tag_diff - MAL_tag_diff) * freq_coeff
                self.user.tag_pos_affinity_dict[tag] = (self.user.tag_pos_affinity_dict[tag] / self.user.tag_counts[tag]) \
                                                       * freq_coeff
            # Add freq_coeff for each tag somewhere
            except ZeroDivisionError:
                self.user.tag_affinity_dict[tag] = 0
                self.user.tag_pos_affinity_dict[tag] = 0 # ?

    def _center_mean_of_affinities(self):
        # freq_multi = len(self.tags.all_anilist_tags)/self.user.sum_of_freq_coeffs
        # change this back later
        def center_tags(tag_type):
            if tag_type == 'Single':
                tags = self.tags.all_single_tags + self.tags.all_genres + self.tags.all_studios
                freq_multi = len(tags) / self.user.sum_of_freq_coeffs_st
                mean_of_affs = np.mean([value for key, value in self.user.tag_affinity_dict.items()
                                        if "<" not in key])
                self.user.mean_of_single_affs = mean_of_affs
                self.user.freq_multi_st = freq_multi
            else:
                tags = self.tags.all_doubletags
                freq_multi = len(tags) / self.user.sum_of_freq_coeffs_dt
                mean_of_affs = np.mean([value for key, value in self.user.tag_affinity_dict.items()
                                        if "<" in key])
                self.user.mean_of_double_affs = mean_of_affs
                self.user.freq_multi_dt = freq_multi
            # mean_of_single_affs = np.mean(list(self.user.tag_affinity_dict.values()))
            # mean_of_single_affs = np.mean([value for key, value in self.user.tag_affinity_dict.items() if "<" not in key])
            # self.user.mean_of_single_affs = mean_of_single_affs

            for tag in tags:
                self.user.tag_affinity_dict[tag] = self.user.tag_affinity_dict[tag] - mean_of_affs\
                                                   * freq_multi * self.user.freq_coeff_per_tag[tag]

        # def center_double_tags():
        #     # freq_multi_dt = len(self.tags.all_doubletags) / self.user.sum_of_freq_coeffs_dt
        #     tags_and_genres = self.tags.all_single_tags + self.tags.all_genres
        #     for tag in tags_and_genres:
        #         non_zero_double_tags = {double_tag : aff for double_tag, aff in self.user.tag_affinity_dict.items()
        #                                        if f"<{tag}>" in double_tag and aff != 0}
        #         sum_of_freq_coeffs = sum([self.user.freq_coeff_per_tag[double_tag]
        #                                   for double_tag in non_zero_double_tags.keys()])
        #         mean_of_double_tags = np.mean(list(non_zero_double_tags.values()))
        #
        #         try:
        #             freq_multi = len(non_zero_double_tags)/sum_of_freq_coeffs
        #         except ZeroDivisionError:
        #             continue  # No need to normalize, they're all 0
        #
        #         for tag in non_zero_double_tags:
        #             self.user.tag_affinity_dict[tag] = self.user.tag_affinity_dict[tag] - mean_of_double_tags\
        #                                                * freq_multi * self.user.freq_coeff_per_tag[tag]

        if self.data.OG_aff_means:
            self.user.center_tag_aff_dict_means(self.data.OG_aff_means)
        center_tags(tag_type='Single')
        center_tags(tag_type='Double')

        # if not tags_to_normalize:
        #     tags_to_normalize = self.tags.all_anilist_tags
        #
        # for tag in tags_to_normalize:
        #     self.user.tag_affinity_dict[tag] = self.user.tag_affinity_dict[tag] - \
        #         self.user.mean_of_affs * self.freq_multi * self.user.freq_coeff_per_tag[tag]
        #
        # for tag in self.tags.all_single_tags:
        #     double_tags = {double_tag: double_tag_value for double_tag, double_tag_value
        #                    in self.user.tag_affinity_dict.items() if f"<{tag}>" in double_tag}
        #     mean_of_double_tags = np.mean(list(double_tags.values()))
        #     for double_tag in double_tags:
        #         self.user.tag_affinity_dict[double_tag] -= mean_of_double_tags

        # if already_normalized:
        #     for tag in tags_to_normalize:
        #         self.user.tag_affinity_dict[tag] -= self.data.OG_aff_means[tag]

    def _process_entry_tags(self, related_entry, length_coeff, user_score, MAL_score):

        # tags = Tags()
        MAL_score_coeff = -0.6 * MAL_score + 5.9

        # related_entry_data = self.tags.entry_tags_dict[related_entry]
        # main_show_data = self.tags.show_tags_dict[main_show]
        related_entry_data = self.tags.entry_tags_dict[related_entry]
        # related_entry_data = self.tags.show_tags_dict[main_show]
        # related_entry_data = adjust_related_entry_data()
        for tag in related_entry_data['Tags'] + related_entry_data['DoubleTags']:
            if tag['name'] not in self.tags.all_anilist_tags:
                continue
            adjusted_p = tag['percentage']
            # adjusted_p = self.tags.adjust_tag_percentage(tag['percentage'])
            # # On Anilist, every tag a show has is listed with a percentage. This percentage
            # # can be pushed down or up a bit by any registered user, so as a variable it
            # # inevitably introduces human error - a tag with 60% means that not all users agree
            # # that said tag is strongly relevant to the show, and thus we want to reduce its weight
            # # by more than just 40% compared to a 100% tag.
            # if adjusted_p == 0:
            #     break

            self.user.score_per_tag[tag['name']] += user_score * adjusted_p * length_coeff
            self.user.MAL_score_per_tag[tag['name']] += MAL_score * adjusted_p * length_coeff
            self.user.tag_counts[tag['name']] += adjusted_p * length_coeff
            self.user.tag_entry_list[tag['name']].append(related_entry)

            if user_score >= self.user.mean_of_watched:
                self.user.tag_pos_affinity_dict[tag['name']] += MAL_score_coeff * \
                                                                self.user.exp_coeffs[10 - user_score] \
                                                                * adjusted_p * length_coeff

        # for genre in self.tags.entry_tags_dict[related_entry]['Genres']:
        for genre in related_entry_data['Genres']:
            self.user.score_per_tag[genre] += user_score * length_coeff
            self.user.MAL_score_per_tag[genre] += MAL_score * length_coeff
            self.user.tag_counts[genre] += length_coeff
            self.user.tag_entry_list[genre].append(related_entry)
            # Genres and studios do not have percentages, so we add 1 as if p=100

            if user_score >= self.user.mean_of_watched:
                self.user.tag_pos_affinity_dict[genre] += MAL_score_coeff * \
                                                          self.user.exp_coeffs[10 - user_score] \
                                                          * length_coeff

        show_studio = self.tags.entry_tags_dict[related_entry]['Studio']
        if show_studio in self.tags.all_anilist_tags:
            self.user.score_per_tag[show_studio] += user_score * length_coeff
            self.user.MAL_score_per_tag[show_studio] += MAL_score * length_coeff
            self.user.tag_counts[show_studio] += length_coeff
            self.user.tag_entry_list[show_studio].append(related_entry)

            if user_score >= self.user.mean_of_watched:
                self.user.tag_pos_affinity_dict[show_studio] += MAL_score_coeff * \
                                                                self.user.exp_coeffs[10 - user_score] \
                                                                * length_coeff

    # def get_user_list_stats(self):
    #     watched_user_score_list = [score for title, score in self.user.scores.items() if score]
    #     # ^ Simply a list of the user's scores for every show they watched
    #
    #     watched_titles = [title for title, score in self.user.scores.items() if score]
    #     # watched_MAL_score_dict = self.data.mean_score_per_show.select(watched_titles).to_dict(as_series=False)
    #     watched_MAL_score_dict = self.data.mean_score_per_show.select(watched_titles).to_dict(as_series = False)
    #     watched_MAL_score_list = [score[0] for title, score in
    #                               watched_MAL_score_dict.items() if title != "Rows"]
    #     self.user.MAL_mean_of_watched = np.mean(watched_MAL_score_list)
    #     self.user.mean_of_watched = np.mean(watched_user_score_list)
    #     self.user.std_of_watched = np.std(watched_user_score_list)

    def get_user_affs(self):

        processed_entries = []
        self.user.entry_list = []
        self.user.show_count = 0

        self.initialize_user_dicts()
        # self.get_user_list_stats()
        self.user = initialize_user_stats(self.user, self.data)

        try:
            x = int(np.ceil(self.user.mean_of_watched))
        except ValueError:
            # This means the user has no watched shows, or there is another issue with the data.
            print(f"Mean is {self.user.mean_of_watched}")
            return self.user

        self.user.score_diffs = [score - self.user.mean_of_watched for score in range(10, x - 1, -1)]
        self.user.exp_coeffs = [1 / 2 ** (10 - exp) * self.user.score_diffs[10 - exp]
                                for exp in range(10, x - 1, -1)]

        # self.user.entry_list.remove('Gin no Saji')
        # self.user.entry_list.remove('Gin no Saji 2nd Season')
        # del self.user.scores['Gin no Saji']
        # del self.user.scores['Gin no Saji 2nd Season']
        # for entry, user_score in self.user.scores.items():
        for entry in self.user.entry_list:
            user_score = self.user.scores[entry]
            if not user_score or entry in processed_entries:
                continue

            main_show = self.tags.entry_tags_dict[entry]['Main']
            # if main_show == 'ReLIFE':
            #     continue
            main_show_data = self.tags.show_tags_dict[main_show]
            user_watched_entries_length_coeffs = [x[1] for x in
                                                  main_show_data['Related'].items()
                                                  if self.user.scores[x[0]]]

            sum_of_length_coeffs = sum(user_watched_entries_length_coeffs)

            for related_entry, length_coeff in main_show_data['Related'].items():
                processed_entries.append(related_entry)
                user_score = self.user.scores[related_entry]
                if not user_score:
                    continue

                length_coeff = length_coeff / sum_of_length_coeffs

                MAL_score = self.data.mean_score_per_show[related_entry][0]
                if MAL_score < 6.5:
                    print("Warning - MAL score lower than 6.5")
                    continue

                # self.user.entry_list.append(related_entry)

                self.user.show_count += length_coeff

                # Calculates sums (user_score_per_show, MAL_score_per_show, etc)
                self._process_entry_tags(related_entry, length_coeff, user_score, MAL_score)

        for tag in self.tags.all_anilist_tags:
            self._calculate_affinities(tag)
        self._center_mean_of_affinities()

        # if self.data.OG_aff_db_means and self.data.OG_aff_db_stds:
        #     for tag in self.user.tag_affinity_dict.keys():
        #         self.user.tag_affinity_dict[tag] = \
        #             (self.user.tag_affinity_dict[tag] - self.data.aff_db_means[tag])/self.data.aff_db_stds[tag]

        return self.user

    def recalc_affinities_2(self, entry_tags_dict, entry):

        def recalc_normal_aff():
            pass

        def recalc_pos_aff():
            pass

        self.user.adj_tag_affinity_dict = self.user.tag_affinity_dict.copy()
        self.user.adj_pos_tag_affinity_dict = self.user.tag_pos_affinity_dict.copy()

        entry_user_score = self.user.scores[entry]
        if not entry_user_score:
            return  # If user hasn't scored the show, there's nothing to recalculate

        main_show = self.tags.entry_tags_dict[entry]['Main']

        length_coeff = self.tags.show_tags_dict[main_show]['Related'][entry]
        sum_of_length_coeffs = sum([length_coeff for related_entry, length_coeff
                                    in self.tags.show_tags_dict[main_show]['Related'].items()
                                    if self.user.scores[entry]])

        entry_MAL_score = self.data.mean_score_per_show[entry].item()
        entry_MAL_coeff = self.MAL_score_coeff(entry_MAL_score)
        # Turn the data thingy into a dict so no .item()
        try:
            entry_exp_coeff = self.user.exp_coeffs[10 - entry_user_score]
        except IndexError:
            entry_exp_coeff = 0

        entry_pos_aff = entry_MAL_coeff * entry_exp_coeff * length_coeff / sum_of_length_coeffs

        for tag, tag_p in entry_tags_dict.items():

            try:
                tag_p = tag_p['percentage']
            except KeyError:
                tag_p = 1

            try:
                tag_overall_ratio =  (self.user.tag_counts[tag]-length_coeff)/(self.user.show_count-length_coeff)
                tag_freq_coeff = min(1, max((self.user.tag_counts[tag] - length_coeff) / 10, tag_overall_ratio * 20))
            except KeyError:
                # Tag is a studio that appears in entry_tags_dict
                # but isn't in tags.all_anilist_tags (too niche)
                continue

            user_score_per_tag = self.user.score_per_tag[tag] - entry_user_score * tag_p
            user_tag_count = self.user.tag_counts[tag] - tag_p

            try:
                user_tag_diff = user_score_per_tag / user_tag_count - self.user.mean_of_watched
            except ZeroDivisionError:
                self.user.adj_tag_affinity_dict[tag] = 0
                self.user.adj_pos_tag_affinity_dict[tag] = 0
                continue  # This entry was the only one with the tag, so affinities are 0 without it

            MAL_score_per_tag = self.user.MAL_score_per_tag[tag] - entry_MAL_score * tag_p
            MAL_tag_diff = MAL_score_per_tag / user_tag_count - self.user.MAL_mean_of_watched

            self.user.adj_tag_affinity_dict[tag] = (2 * user_tag_diff - MAL_tag_diff) * tag_freq_coeff
            self.user.adj_tag_affinity_dict[tag] -= self.data.OG_aff_means[tag]

            mean_of_affs = self.user.mean_of_double_affs if "<" in tag else self.user.mean_of_single_affs
            freq_multi = self.user.freq_multi_dt if "<" in tag else self.user.freq_multi_st
            self.user.adj_tag_affinity_dict[tag] -= mean_of_affs * freq_multi * self.user.freq_coeff_per_tag[tag]

            entry_pos_aff /= self.user.tag_counts[tag] # do minus here?
            entry_pos_aff *= self.user.freq_coeff_per_tag[tag]

            self.user.adj_pos_tag_affinity_dict[tag] -= entry_pos_aff
            self.user.adj_pos_tag_affinity_dict[tag] *= (tag_freq_coeff / self.user.freq_coeff_per_tag[tag])
            self.user.adj_pos_tag_affinity_dict[tag] *= (self.user.tag_counts[tag] / user_tag_count)




            # deal with singles and doubles in norm (check for < then substract appropriate mean?)


    def recalc_affinities_without_entry(self,entry):
        tags_to_recalc = self.tags.show_tags_dict[entry]['Tags'] + \
                         self.tags.show_tags_dict[entry]['Genres'] \

        for related_entry, studio in self.tags.show_tags_dict[entry]['Studios'].items():
            if studio not in tags_to_recalc and studio in self.tags.all_anilist_tags \
                    and self.user.scores[related_entry]:
                tags_to_recalc.append(studio)


        # tag_percentages = [x['percentage'] for x in self.tags.show_tags_dict[entry]['Tags']]
        # tag_names = [tag['name'] for tag in tags_to_recalc]
        # tag_percentages = [tag['percentage'] for tag in tags_to_recalc]
        self.user.adj_tag_affinity_dict = self.user.tag_affinity_dict.copy()
        self.user.adj_pos_tag_affinity_dict = self.user.tag_pos_affinity_dict.copy()
        sum_of_length_coeffs = sum([length_coeff for related_entry, length_coeff
                                        in self.tags.show_tags_dict[entry]['Related'].items()
                                        if self.user.scores[entry]])
        entry_pos_aff = sum([self.user.exp_coeffs[10 - self.user.scores[related_entry]] *
                             self.MAL_score_coeff(self.data.mean_score_per_show[related_entry].item()) *
                             length_coeff / sum_of_length_coeffs
                             for related_entry, length_coeff
                             in self.tags.show_tags_dict[entry]['Related'].items()
                             if self.user.scores[related_entry] and
                             self.user.scores[related_entry] > self.user.mean_of_watched])
        for tag in tags_to_recalc:
            if not (tag in self.tags.all_anilist_tags or tag['name'] in self.tags.all_anilist_tags):
                continue
            self._calculate_affinities(tag, recalc=True, show_to_exclude=entry,
                                       entry_pos_aff = entry_pos_aff, sum_of_length_coeffs=sum_of_length_coeffs)
        # self._center_mean_of_affinities(already_normalized=True, tags_to_normalize=tag_names)

    @staticmethod
    def MAL_score_coeff(score):
        return -0.6 * score + 5.9


#  ------------------------------------------------------------------------------------------------
class AffinityDB:
    _instance = None

    minor_parts_per_process = 500
    size_limit = 20_000_000

    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, db_type=None):
        # All properties are loaded on demand
        self._df = None
        self._data = None
        # self.size_limit = 20_000_000
        self.features = self.Features()
        # self.users_tag_affinity_dict = {}
        self._OG_aff_means = None
        self._major_parts = None
        self._db_type = db_type
        # self.anime_df = AnimeDB()
        # self.tags = Tags()
        # self.graphs = Graphs() #remove the selfs from other classes too?...

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
                self._data = load_pickled_file("general_data.pickle")
            except FileNotFoundError:
                print("GeneralData not found, proceeding to load")
                self._data = GeneralData()
                self._data = self._data.generate_data()
                # self._get_general_data()
                save_pickled_file("general_data.pickle", self._data)
        return self._data

    # @property
    # def df(self):
    #     """Polars database (stored as .parquet), contains username
    #     + mean score + scored shows + all the scores of each user in it."""
    #     if not (isinstance(self._df, pl.DataFrame) or
    #             isinstance(self._df, pd.DataFrame)):
    #         try:
    #             print("Loading affinity database")
    #             self._df = pl.read_parquet(f"{aff_db_filename}.parquet")
    #         except FileNotFoundError:
    #             print("Affinity database not found. Creating new affinity database")
    #             self._create()
    #             self.combine()
    #             self.shuffle()
    #     return self._df

    @property
    def major_parts(self):
        if self._major_parts is None:
            self._major_parts = self.count_major_parts()
        return self._major_parts

    @property
    def db_type(self):
        if not self._db_type:
            try:
                df = pq.ParquetFile(f"{aff_db_filename}.parquet")
                features = df.schema.names
                if "Max Affinity" in features:
                    self._db_type = 2
                else:
                    self._db_type = 1
            except FileNotFoundError:
                self._db_type = 1
        return self._db_type





    # @staticmethod
    # def _load_affinity_DB(parts):
    #     print(f"Unpacking part 1 of database")
    #     df = pl.read_parquet(f"Partials\\{aff_db_filename}-PP1.parquet")
    #     for i in range(2, parts + 1):
    #         try:
    #             print(f"Unpacking part {i} of database")
    #             temp_df = pl.read_parquet(f"Partials\\{aff_db_filename}-PP{i}.parquet")
    #             df = df.vstack(temp_df)
    #         except FileNotFoundError:
    #             break
    #     return df
    #
    # def _load_affinity_DB2(self, parts):
    #     print(f"Unpacking part 1 of database")
    #     df = pd.read_parquet(f"Partials\\{aff_db_filename}-PP1.parquet")
    #     for i in range(2, parts + 1):
    #         try:
    #             print(f"Unpacking part {i} of database")
    #             temp_df = pd.read_parquet(f"Partials\\{aff_db_filename}-PP{i}.parquet")
    #             df = pd.concat([df, temp_df], ignore_index=True)
    #         except FileNotFoundError:
    #             break
    #         if df.shape[0] > self.size_limit:
    #             return df[0:self.size_limit]
    #     return df
    @timeit
    def create(self):
        parts = int(os.cpu_count() / 4)
        if not os.path.exists(f"Partials\\UserDB-P{parts}.parquet"):
            user_db = UserDB()
            user_db.split_df(parts)
            del user_db

        if not os.path.isfile("general_data.pickle"):
            data = GeneralData()
            data.generate_data()
            save_pickled_file("general_data.pickle", data)

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
        save_data_per = user_amount // self.minor_parts_per_process
        t1 = time.perf_counter()

        for user_index in range(user_amount):
            if user_index % 10 == 0:
                print(general_utils.time_at_current_point(t1))
                print(f"Currently on user {user_amount * i + user_index}")

            user = User(name=partial_main_df['Username'][user_index],
                        scores=partial_main_df.select(self.data.relevant_shows).row(user_index, named=True),
                        scored_amount = partial_main_df["Scored Shows"][user_index])

            user = initialize_user_stats(user, self.data)

            aff_db_entry_creator = AffDBEntryCreator(user, self.data, tags)
            aff_db_entry_creator.create_db_entries_from_user_list(db_type=1, for_predict=False)

            user_db_entries = aff_db_entry_creator.aff_db_entry_dict

            for stat, stat_values in user_db_entries.items():
                # We concatenate the existing db values with the ones returned from aff_db_entry_creator
                aff_db_dict[stat] = aff_db_dict[stat] + stat_values

            if (user_index + 1) % save_data_per == 0:
                subpart_num = (user_index + 1) // save_data_per
                print(f"Finished processing user {self.data.user_amount * i + user_index + 1},"
                      f" saving PP-{self.minor_parts_per_process * i + subpart_num}")
                # save_pickled_file(f"user_tag_affinity_dict-PP{subpart_num}.pickle", user_tag_affinity_dict)
                # Fix the above (either remove this entirely or concatenate them at the end?)

                for key in aff_db_dict.keys():
                    aff_db_dict[key] = np.array(aff_db_dict[key], dtype=np.float32)

                pl.DataFrame(aff_db_dict).write_parquet(
                    f"Partials\\{aff_db_filename}-PP{self.minor_parts_per_process * i + subpart_num}.parquet")


                # Each batch of save_data_per users will be separated into their own mini-database during
                # runtime to avoid blowing up my poor 32GB of RAM

                # After saving the data, we need to reinitialize the dicts to avoid wasting memory

                aff_db_dict = AffDBEntryCreator.initialize_aff_db_dict(db_type=1)

    # def get_user_list_stats(user: User, data: Data):
    #     watched_user_score_list = [score for title, score in user.scores.items() if score]
    #     # ^ Simply a list of the user's scores for every show they watched
    #
    #     watched_titles = [title for title, score in user.scores.items() if score]
    #     # watched_MAL_score_dict = self.data.mean_score_per_show.select(watched_titles).to_dict(as_series=False)
    #     watched_MAL_score_dict = data.mean_score_per_show.select(watched_titles).to_dict(as_series=False)
    #     watched_MAL_score_list = [score[0] for title, score in
    #                               watched_MAL_score_dict.items() if title != "Rows"]
    #     user.MAL_mean_of_watched = np.mean(watched_MAL_score_list)
    #     user.mean_of_watched = np.mean(watched_user_score_list)
    #     user.std_of_watched = np.std(watched_user_score_list)
    #     return user

    @staticmethod
    def combine(max_examples_per_df=1_500_000):
        """Takes the small chunks created by create_affinity_DB and turns them into larger
        chunks, each of which is a separate database of ~1,500,000 records."""
        print(f"Unpacking minor chunk 1 of database")
        j = 1
        df = pl.read_parquet(f"Partials\\{aff_db_filename}-PP1.parquet")
        for i in range(2, 100000):
            try:
                print(f"Unpacking minor chunk {i} of database")
                temp_df = pl.read_parquet(f"Partials\\{aff_db_filename}-PP{i}.parquet")
                df = df.vstack(temp_df)
            except FileNotFoundError:
                df.write_parquet(f"Partials\\{aff_db_filename}-P{j}.parquet")
                break
            if len(df) > max_examples_per_df:
                print(f"Reached {max_examples_per_df} examples. Saving part {j}")
                df.write_parquet(f"Partials\\{aff_db_filename}-P{j}.parquet")
                j += 1
                df = pl.DataFrame()

    def shuffle(self):
        p = self.major_parts
        print(f"Total major part amount : {p}")

        # First we shuffle each major database chunk. This isn't enough for a thorough
        # shuffle - since each major chunk is currently simply a combination of

        # Then we separate each major chunk into major_part_amount smaller chunks.
        for i in range(p):
            print(f"---------Loading major part {i + 1} of database----------")
            df = pd.read_parquet(f"Partials\\{aff_db_filename}-P{i + 1}.parquet")
            df = shuffle_df(df)
            sub_size = int(len(df) / p)
            for j in range(p):
                print(f"--Writing minor shuffled part {p * i + j + 1} of database--")
                df[j * sub_size:(j + 1) * sub_size].to_parquet(
                    f"Partials\\{aff_db_filename}-PP{p * i + j + 1}.parquet")

        print("Finished shuffling all major parts. Beginning to recombine.")

        for i in range(p):
            df = pd.DataFrame()
            for j in range(i, p * p, p):
                print(f"--Loading minor shuffled part {j + 1} of database--")
                chunk = pd.read_parquet(f"Partials\\{aff_db_filename}-PP{j + 1}.parquet")
                df = pd.concat([df, chunk], ignore_index=True)

            print(f"---------Writing major shuffled part {i + 1} of database----------")
            df.to_parquet(f"Partials\\{aff_db_filename}-P{i + 1}.parquet")

    def normalize(self):
        amount_of_dfs = AffinityDB.count_major_parts()
        # data = load_pickled_file("general_data.pickle")
        aff_means = None
        aff_stds = None
        for i in range(amount_of_dfs):
            print(f"Normalizing chunk {i + 1}")
            df = pd.read_parquet(f"Partials\\{aff_db_filename}-P{i + 1}.parquet")
            df = self.normalize_aff_df(df, is_aff_db=True)
            df.to_parquet(f"Partials\\{aff_db_filename}-P{i + 1}-N.parquet")
            if aff_means and aff_stds:
                for col in aff_means.keys():
                    aff_means[col] = (self.data.aff_means[col] + aff_means[col] * i) / (i + 1)
                    aff_stds[col] = (self.data.aff_stds[col] + aff_stds[col] * i) / (i + 1)
            else:
                aff_means = self.data.aff_means
                aff_stds = self.data.aff_stds

        self.data.aff_means = aff_means
        self.data.aff_stds = aff_stds
        save_pickled_file("general_data.pickle", self.data)

    def normalize_aff_df(self,df=None, og_filename=None, for_predict=False, is_aff_db=False):
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
            if type(self.data) != GeneralData:
                raise ValueError("Data must be sent to acquire mean of main affinity database"
                                 "if for_predict=True")
            for col in cols_for_norm:
                df.loc[df[col] != 0, col] = \
                    (df.loc[df[col] != 0, col] - self.data.aff_means[col]) / self.data.aff_stds[col]
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
            self.data.aff_means = mean_dict
            self.data.aff_stds = std_dict
            save_pickled_file("general_data.pickle", self.data)

        return df

    @staticmethod
    def remove_show_score():
        amount_of_dfs = AffinityDB.count_major_parts()
        for i in range(amount_of_dfs):
            print(f"Removing means from normalized chunk {i + 1}")
            df = pd.read_parquet(f"Partials\\{aff_db_filename}-P{i + 1}-N.parquet")
            # df.drop('Show Score',inplace=True, axis=1)
            # for j in range(1,31):
            #     for stat in ["Max Affinity", "Min Affinity", "Avg Affinity", "Max Pos Affinity"]:
            #         df.drop(f"Doubles-{j} {stat}", inplace=True, axis=1)
            cols_to_take = [x for x in df.columns if not x.startswith("Doubles-")
                            and x!= 'Show Score']
            df = df[cols_to_take]
            df.to_parquet(f"Partials\\{aff_db_filename}-P{i + 1}-N-RSDD.parquet")

    # @staticmethod
    # def remove_extra_doubles():
    #     amount_of_dfs = AffinityDB.count_major_parts()
    #     for i in range(amount_of_dfs):
    #         df = pd.read_parquet(f"Partials\\{aff_db_filename}-P{i + 1}-N.parquet")

    @staticmethod
    def count_major_parts():
        p = 1
        while True:
            if not os.path.exists(f"Partials\\{aff_db_filename}-P{p}.parquet"):
                break
            p += 1
        return p - 1

