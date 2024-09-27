from dataclasses import dataclass, field
import polars as pl
from .Tags import Tags
from .AnimeDB import AnimeDB
from .UserDB import UserDB

# Works as is for now, try to remove this mess later
# without losing the speed boost


@dataclass
class GeneralData:
    """"""
    relevant_shows: list = field(default=None)

    mean_score_per_show: pl.DataFrame = field(default=None)
    scored_members_per_show: pl.DataFrame = field(default=None)
    scored_shows_per_user: pl.Series = field(default=None)

    user_amount: int = field(default=None)

    aff_means: dict = field(default=None)
    aff_stds: dict = field(default=None)

    OG_aff_means: dict = field(default=None)
    OG_aff_stds: dict = field(default=None)

    def generate_data(self):
        def get_mean_score_per_show(anime_df):
            mean_score_per_show = anime_df.filter(pl.col('Rows') == "Mean Score").to_dict(as_series=False)
            del mean_score_per_show['Rows']
            mean_score_per_show = {show: mean_score[0] for show, mean_score in mean_score_per_show.items()}
            return mean_score_per_show

        tags = Tags()
        anime_db = AnimeDB()
        user_db = UserDB()
        self.relevant_shows = list(tags.entry_tags_dict_nls.keys())
        partial_main_df = user_db.df.select(user_db.stats + self.relevant_shows)
        partial_anime_df = anime_db.df.select(["Rows"] + self.relevant_shows)

        self.mean_score_per_show_full = get_mean_score_per_show(anime_db.df)
        self.mean_score_per_show = get_mean_score_per_show(partial_anime_df)
        # For usage comfort, to avoid filtering every time we need the partial one

        # del self.mean_score_per_show['Rows']
        # self.mean_score_per_show = {show: mean_score[0] for show, mean_score in self.mean_score_per_show.items()}

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
