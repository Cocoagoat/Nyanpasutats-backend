from dataclasses import dataclass, field
import polars as pl
import pandas as pd
from .Tags import Tags
from .AnimeDB import AnimeDB
from .UserDB import UserDB
from .filenames import *
import os
from .User import User


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
        # if os.path.isfile(aff_db_filename):
        #     self._get_means_of_affs() # This is done when normalizing, can't do it before we actually get the DB...
        #     self._get_db_type()
        # self._get_means_of_OG_affs(partial_main_df)

        return self
        # self._get_means_of_OG_aff_columns()


    # def _get_db_type(self):
    #     pass

    # def _get_means_of_affs(self):
    #     aff_db = AffinityDB()
    #     mean_dict = {}
    #     std_dict = {}
    #
    #     if aff_db.df is not None: # Think about how to include those means later? Also why would aff_db.df be equal none
    #         cols_for_norm = [x for x in list(aff_db.df.columns) if x not in
    #                          ["Pos Affinity Ratio", "Neg Affinity Ratio", "Show Popularity", "User Scored Shows",
    #                           "User Score"]]
    #         for col in cols_for_norm:
    #             # mean_dict[col] = aff_db.df.filter(aff_db.df[col] != 0)[col].mean()
    #             # std_dict[col] = aff_db.df.filter(aff_db.df[col] != 0)[col].std()
    #             mean_dict[col] = aff_db.df[col].mean()
    #             std_dict[col] = aff_db.df[col].std()
    #         self.aff_means = mean_dict
    #         self.aff_stds = std_dict

    # def _get_means_of_OG_affs(self, partial_main_df):
    #     tags = Tags()
    #     users_tag_affinity_dict = {}
    #     for user_index in range(0, self.user_amount, 10):
    #         if user_index%100 == 0:
    #             print(f"Calculating affinities of each user to each tag,"
    #                    f" currently on user {user_index} out of {self.user_amount}")
    #         user = User(name=partial_main_df['Username'][user_index],
    #                     scores=partial_main_df.select(self.relevant_shows).row(user_index, named=True),
    #                     scored_amount=partial_main_df["Scored Shows"][user_index])
    #         user_aff_calculator = UserAffinityCalculator(user, self, tags)
    #         user = user_aff_calculator.get_user_affs()
    #         if not user.tag_affinity_dict:
    #             continue
    #         users_tag_affinity_dict[user.name] = user.tag_affinity_dict
    #
    #     OG_tag_aff_db = pd.DataFrame(users_tag_affinity_dict)
    #     OG_tag_aff_db.to_parquet(data_path / "OG_tag_aff_db.parquet")
    #
    #     self.OG_aff_means = {}
    #     for i,col in enumerate(OG_tag_aff_db.index):
    #         self.OG_aff_means[col] = OG_tag_aff_db.iloc[i].mean()