import numpy as np
import polars as pl
from main.modules.filenames import anime_database_updated_name
from main.modules.AnimeDB import AnimeDB
from main.modules.AnimeListFormatter import ListFormatter
from main.modules.AnimeListHandler import AnimeListHandler
from main.modules.MAL_utils import *
from main.models import AnimeData, AnimeDataUpdated
from main.modules.Tags import Tags
from main.modules.general_utils import snake_to_camel, timeit
from collections import defaultdict
from scipy import stats as scipy_stats
from main.modules.GlobalValues import SUPPORTED_WEBSITES
from main.modules.django_db_utils import get_objects_preserving_order


class Season:
    def __init__(self, name=""):
        self.name = name
        self.shows = 0
        self.scored_shows = 0
        self.dropped_shows = 0
        self.total_score = 0
        self.avg_score = 0
        self.favorites_avg_score = 0
        self.overall_rank = None
        self.yearly_rank = None
        self.most_watched_genre = None
        self.most_unusual_show = None
        self.affinity = None
        self.total_shows_duration = 0
        self.favorites_rank = None
        self.fav_yearly_rank = None
        self.__show_list_sorted = False
        self.show_list = {}

    def add_show(self, show_name, user_score, user_list_status):
        self.shows += 1
        if user_list_status == 'dropped':
            self.dropped_shows += 1
        if user_score:
            self.total_score += user_score
            self.scored_shows += 1
            self.show_list[show_name] = user_score

    def calculate_avg_score(self):
        return self.total_score / self.scored_shows if self.scored_shows else 0

    def get_db_objects_of_shows(self):
        return get_objects_preserving_order(AnimeDataUpdated,
                                            self.show_list.keys(), 'name')

    def sort_show_list(self):
        self.show_list = {show: score for show, score in sorted(
                self.show_list.items(), reverse=True, key=lambda x: x[1])}

    @staticmethod
    def get_season_value(season_name):
        season_name, year = season_name.split(" ")
        return int(year) + (Seasons[season_name].value/10)


class SeasonCollection:
    def __init__(self):
        self.seasons = {}
        self.__sorted = False

    def __iter__(self):
        return iter(self.seasons.items())

    @timeit
    def to_dict(self):
        seasons = {}
        excluded_attrs = ["total_score", "scored_shows", "name"]
        for season_name, season_stats in self:
            season_dict = {}
            for attr, attr_value in season_stats.__dict__.items():
                if attr.startswith("_") or attr in excluded_attrs:
                    continue
                camel_case_attr = snake_to_camel(attr)
                season_dict[camel_case_attr] = attr_value
            seasons[season_name] = season_dict
        return seasons

    def get_season(self, name):
        return self.seasons.setdefault(name, Season(name))

    def add_show_to_season(self, season_name, show_name,
                           user_score, user_list_status):

        season = self.get_season(season_name)
        season.add_show(show_name, user_score, user_list_status)

    def sort_seasons_by_avg_score(self):
        self.__sorted = True
        self.seasons = {season: season_stats for season, season_stats
                        in sorted(self.seasons.items(),
                        key=lambda x: x[1].avg_score,
                        reverse=True)}

    def sort_seasons_by_fav_avg_score(self):
        self.__sorted = True
        self.seasons = {season: season_stats for season, season_stats
                        in sorted(self.seasons.items(),
                        key=lambda x: x[1].favorites_avg_score,
                        reverse=True)}

    def sort_seasons_by_attribute(self, attribute_name):
        if not hasattr(Season(), attribute_name):
            raise ValueError(f"Attribute {attribute_name} not found in Season class.")

        self.__sorted = True
        if not attribute_name == "name":
            sort_key = lambda x: getattr(x[1], attribute_name)

        else:
            sort_key = lambda x: Season.get_season_value(getattr(x[1], attribute_name))

        self.seasons = {season: season_stats for season, season_stats
                        in sorted(self.seasons.items(),
                                  key=sort_key,
                                  reverse=True)}

    def sort_show_lists(self):
        for season_name, season_stats in self.seasons.items():
            season_stats.sort_show_list()


class SeasonalStats2:
    STAT_NAMES = ["avg_score", "overall_rank", "most_watched_genre",
                  "most_unusual_show", "affinity", "favorites_avg_score",
                  "total_shows_duration", "favorites_rank", "yearly_rank",
                  "fav_yearly_rank"]

    def __init__(self, username, site, no_sequels=False, min_shows=5):
        self.username = username
        self.no_sequels = no_sequels
        if site in SUPPORTED_WEBSITES:
            self.site = site
        else:
            raise ValueError(f"{site} is not a website supported by this application.")
        self.min_shows = min_shows
        self._full_stats = SeasonCollection()
        self._full_stats_sorted_by_fav_avg = SeasonCollection()
        self._full_stats_no_sequels = SeasonCollection()
        self._full_stats_sorted_by_fav_avg_no_sequels = SeasonCollection()
        self.user_list = None
        self.anime_data = None
        self.tags = Tags()
        self.anime_db = pl.read_parquet(anime_database_updated_name)

    @property
    def full_stats(self):
        if not self._full_stats.seasons:
            self.get_user_seasonal_stats2()
        return self._full_stats

    @property
    def full_stats_sorted_by_fav_avg(self):
        if not self._full_stats_sorted_by_fav_avg.seasons:
            self.sort_full_stats_by_fav_avg()
        return self._full_stats_sorted_by_fav_avg

    def __initialize_seasonal_stats(self):
        for anime in self.anime_data:
            # test = self.anime_data
            # if not anime.type:
            #     print("ERROR")
            if anime.episodes >= 6 or ((anime.year == datetime.datetime.now().year) and (
                    anime.season == (datetime.datetime.now().month-1)//3 + 1) and anime.type and (
                    int(anime.type) == 1)):  # OVAs or movies don't count as part of the season
                try:
                    season_name = Seasons(anime.season).name  # Seasons are listed as 1,2,3,4 in the database
                except AttributeError:
                    continue

                user_score = self.user_list[anime.name]['score']
                user_list_status = self.user_list[anime.name]['list_status']

                full_season_name = f"{season_name} {anime.year}"
                if not full_season_name:
                    continue  # remove?
                self._full_stats.add_show_to_season(full_season_name, anime.name,
                                                    user_score, user_list_status)

    @staticmethod
    def filter_by_min_shows(full_stats: SeasonCollection):
        full_stats.seasons = {season_name: season_data for season_name, season_data
                              in full_stats.seasons.items() if season_data.scored_shows >= 5}
        return full_stats

    @timeit
    def sort_full_stats_by_fav_avg(self):
        self._full_stats_sorted_by_fav_avg.seasons = self._full_stats.seasons.copy()
        self._full_stats_sorted_by_fav_avg.sort_seasons_by_attribute("favorites_avg_score")

    @timeit
    def _add_stat(self, stat_name):
        print(f"Stat name is {stat_name}")
        if stat_name not in self.STAT_NAMES:
            raise ValueError(f"{stat_name} is not a valid seasonal stat")

        elif stat_name == "avg_score":
            self._add_avg_score()
        elif stat_name in ["overall_rank", "favorites_rank"]:
            self._add_rank(stat_name)
        elif stat_name in ["yearly_rank", "fav_yearly_rank"]:
            self._add_yearly_rank(stat_name)
        elif stat_name == "most_watched_genre":
            self._add_most_watched_genre()
        elif stat_name == "most_unusual_show":
            self._add_most_unusual_show()
        elif stat_name == "affinity":
            self._add_affinity()
        elif stat_name == "favorites_avg_score":
            self._add_fav_avg_score()
        elif stat_name == "total_shows_duration":
            self._add_total_shows_duration()

    def _add_avg_score(self):
        for season_name, season_stats in self._full_stats:
            season_stats.avg_score = season_stats.calculate_avg_score()

    def _add_fav_avg_score(self):
        for season_name, season_stats in self._full_stats:
            if season_stats.shows >= 10:
                scores_of_favorites = list(season_stats.show_list.values())[0:10]
                sum_of_favorites = sum((score for score in scores_of_favorites))
                fav_avg_score = sum_of_favorites / 10
            else:
                fav_avg_score = season_stats.avg_score

            season_stats.favorites_avg_score = fav_avg_score

    @timeit
    def _add_rank(self, rank_type):
        valid_rank_types = ["overall_rank", "favorites_rank"]
        if rank_type not in valid_rank_types:
            raise ValueError(f"Unknown rank type. The only allowed rank types are {valid_rank_types}")
        avg_type = "avg_score" if rank_type == "overall_rank" else "favorites_avg_score"
        stats = self._full_stats if rank_type == "overall_rank" else self.full_stats_sorted_by_fav_avg

        prev_season_name = None
        for i, (season_name, season_stats) in enumerate(stats, start=1):
            if i == 1:
                setattr(season_stats, rank_type, 1)
            else:
                prev_season_stats = stats.get_season(prev_season_name)
                if getattr(prev_season_stats, avg_type) == getattr(season_stats, avg_type):
                    prev_season_rank = getattr(prev_season_stats, rank_type)
                    setattr(season_stats, rank_type, prev_season_rank)
                    # season_stats.overall_rank = prev_season_stats.overall_rank
                else:
                    setattr(season_stats, rank_type, i)
            prev_season_name = season_name

    def _add_yearly_rank(self, rank_type):
        valid_rank_types = ["yearly_rank", "fav_yearly_rank"]
        if rank_type not in valid_rank_types:
            raise ValueError(f"Unknown yearly rank type. "
                             f"The only allowed rank types are {valid_rank_types}")

        # avg_type = "fav_avg_score" if rank_type == "overall_rank" else "favorites_avg_score"
        stats = self._full_stats if rank_type == "yearly_rank" else self.full_stats_sorted_by_fav_avg

        # current_year = datetime.date.today().year
        # years_counter = {str(year): 0 for year in range(1960, current_year + 1)}
        years_counter = defaultdict(int)
        for season_name, season_stats in stats:
            season, year = season_name.split(" ")
            setattr(season_stats, rank_type, years_counter[year]+1)
            # season_stats.yearly_rank = years_counter[year]+1
            years_counter[year] += 1

    def _add_most_watched_genre(self):
        for season_name, season_stats in self._full_stats:
            genre_counts = defaultdict(int)
            for show in season_stats.show_list:
                try:
                    show_genres = self.tags.entry_tags_dict[show]['Genres']
                except KeyError:
                    continue

                for genre in show_genres:
                    genre_counts[genre] += 1

            try:
                most_watched_genre = max(genre_counts, key=genre_counts.get)
            except ValueError:
                most_watched_genre = None
            season_stats.most_watched_genre = most_watched_genre

    def _add_most_unusual_show(self, no_unusual_found=False):
        def is_unusual(MAL_score, score):
            unusual_conditions = [MAL_score < 6 and score >= 7,
                                        MAL_score < 7 and score >= 8,
                                        MAL_score < 7.5 and score >= 9,
                                        MAL_score < 8 and score == 10,
                                        MAL_score > 9 and score <= 7,
                                        MAL_score > 8 and score <= 6,
                                        MAL_score > 7.5 and score <= 5,
                                        MAL_score > 7 and score <= 4]
            return any(unusual_conditions)

        def find_most_unusual_in_season(show_list, no_unusual_found=False):
            max_diff = 0
            for show, score in show_list.items():
                if score == 0:
                    continue
                MAL_score = MAL_scores[show]
                diff = abs(score - MAL_score)

                if diff > max_diff and (is_unusual(MAL_score, score) or no_unusual_found):
                    max_diff = diff
                    max_diff_show = show

            if max_diff == 0:
                # No show that matches unusual_conditions was found, in that case we'll return
                # the show with the biggest score difference between the user's score and the show's
                # MAL score.
                max_diff_show = find_most_unusual_in_season(show_list, no_unusual_found=True)

            return max_diff_show

        for season_name, season_stats in self._full_stats:

            # MAL_scores = {anime.name: float(anime.mean_score)
            #               for anime in self.anime_data.filter(
            #               name__in=season_stats.show_list.keys())}
            # ^ do not do this, every filter is really slow

            MAL_scores = {anime.name: float(anime.mean_score) for anime in
                          self.anime_data if anime.name in season_stats.show_list.keys()}

            most_unusual_score = find_most_unusual_in_season(season_stats.show_list)
            # for show, score in season_stats.show_list.items():
            #     if score == 0:
            #         continue
            #     MAL_score = MAL_scores[show]
            #     diff = abs(score - MAL_score)
            #
            #     if diff > max_diff and (is_unusual(MAL_score, score) or no_unusual_found):
            #         max_diff = diff
            #         max_diff_show = show
            #
            # if max_diff == 0:
            #     # No show that matches unusual_conditions was found, in that case we'll return
            #     # the show with the biggest score difference between the user's score and the show's
            #     # MAL score.
            #     max_diff_show = self._add_most_unusual_show(no_unusual_found=True)
            season_stats.most_unusual_show = most_unusual_score

    def _add_affinity(self):
        # anime_db = AnimeDB()
        for season_name, season_stats in self._full_stats:
            user_show_list = season_stats.show_list
            user_scores = np.array([user_show_list[title]
                                    for title in user_show_list.keys()])

            # MAL_scores = {anime.name: float(anime.mean_score)
            #               for anime in self.anime_data.filter(
            #         name__in=user_show_list.keys())}

            # user_show_objects = get_objects_preserving_order(
            #     AnimeDataUpdated, user_show_list.keys(), 'name')

            user_seasonal_shows = self.anime_db.select(user_show_list.keys())
            MAL_scores = np.array(user_seasonal_shows[AnimeDB.stats["Mean Score"]].row(0))
            # MAL_scores = np.array([float(anime.mean_score) if anime else 0
            #              for anime in user_show_objects])

            non_zero_mask = (user_scores != 0) & (MAL_scores != 0)

            user_scores = user_scores[non_zero_mask]
            MAL_scores = MAL_scores[non_zero_mask]
            affinity = np.corrcoef(user_scores, MAL_scores)[0][1]
            season_stats.affinity = affinity if not np.isnan(affinity) else 0

    def _add_total_shows_duration(self):
        for season_name, season_stats in self._full_stats:
            # show_obj_list = [AnimeData.objects.get(name=title) for title in season_stats.show_list]
            # show_obj_list = season_stats.get_db_objects_of_shows()

            season_stats.total_shows_duration = float(sum([show.duration *
                                                           self.user_list[show.name]['num_watched']
                                                           for show in self.anime_data if show.name
                                                           in season_stats.show_list.keys()]))

    # def _add_favorites_rank(self):
    #     for season_name, season_stats in self._full_stats_sorted_by_fav_avg:

    @timeit
    def get_user_seasonal_stats2(self):
        # Main function
        tags = Tags()
        ListHandler = AnimeListHandler.get_concrete_handler(self.site)
        self.user_list = ListHandler(self.username).anime_list.list
        self.user_list = {show: stats for show, stats in self.user_list.items()
                          if stats['list_status'] == "dropped" or stats['score']}

        if self.no_sequels:
            # Filtering out sequels
            self.user_list = {title: stats for title, stats
                              in self.user_list.items()
                              if title in tags.show_tags_dict.keys()}
            # or title in new anime db but not in old anime db?

        self.anime_data = AnimeDataUpdated.objects.filter(name__in=self.user_list.keys())
        # The SQLite DB object with data on all anime

        self.__initialize_seasonal_stats()
        test = self._full_stats
        self._full_stats = self.filter_by_min_shows(self._full_stats)
        self._full_stats.sort_show_lists()

        self._add_stat("avg_score")
        self._full_stats.sort_seasons_by_attribute("avg_score")

        for stat in SeasonalStats2.STAT_NAMES:
            self._add_stat(stat)
        self._full_stats.sort_seasons_by_attribute("name")


        # self._add_stat("overall_rank")
        # self._add_stat("most_watched_genre")
        # self._add_stat("most_unusual_show")
        # self._add_stat("affinity")
        # self._add_stat('favorites_avg_score')
        # self._add_stat("total_shows_duration")
        # self._add_stat("favorites_rank")
        # self._add_stat("yearly_rank")
        # self._add_stat("fav_yearly_rank")
        # self._add_stat("")
        # return self._full_stats

        # prev_season = None
        # for i, (season, season_stats) in enumerate(sorted_seasonal_stats.items(), start=1):
        #
        #     sorted_seasonal_stats[season]['OverallRank'] = get_season_rank(sorted_seasonal_stats,
        #                                                                    prev_season, season, i)
        #
        #     sorted_seasonal_stats[season]['MostWatchedGenre'] = get_most_watched_genre(season_stats['ShowList'])
        #     sorted_seasonal_stats[season]['MostControversialShow'] = find_most_controversial_score_in_season(season_stats)
        #     sorted_seasonal_stats[season]['Affinity'] = calculate_contrarian_index(user_list, season_stats['ShowList'])
        #     sorted_seasonal_stats[season]['ShowList'] = sort_season_shows_by_scores(seasonal_dict[season]['ShowList'])
        #     sorted_seasonal_stats[season]['FavoritesAvgScore'] = calculate_favs_avg_score(
        #         sorted_seasonal_stats, season)
        #     sorted_seasonal_stats[season]['TotalShowsDuration'] = get_total_shows_duration(seasonal_dict[season]['ShowList'],
        #                                                                                    user_list)
        #     prev_season = season
        #
        #
        # sorted_by_favs_seasonal_stats = {season: stats for season, stats in sorted(sorted_seasonal_stats.items(),
        #                                                                    key=lambda x: x[1]['FavoritesAvgScore'],
        #                                                                    reverse=True)}
        #
        # prev_season = None
        # for i, (season, _) in enumerate(sorted_by_favs_seasonal_stats.items(), start=1):
        #     sorted_seasonal_stats[season]['FavoritesRank'] = get_season_rank(sorted_by_favs_seasonal_stats,
        #                                                                      prev_season, season, i, favorites=True)
        #     prev_season = season
        #
        # current_year = datetime.date.today().year
        #
        # years_counter = {str(year): 0 for year in range(1960, current_year+1)}
        #
        # for season, season_stats in sorted_seasonal_stats.items():
        #     season_name, year = season.split(" ")
        #     season_stats['YearlyRank'] = years_counter[year] + 1
        #     years_counter[year] += 1
        #
        # years_counter = {str(year): 0 for year in range(1960, current_year+1)}
        #
        # for season, season_stats in sorted_by_favs_seasonal_stats.items():
        #     season_name, year = season.split(" ")
        #     season_stats['FavYearlyRank'] = years_counter[year] + 1
        #     years_counter[year] += 1
        #
        # sorted_seasonal_stats = sort_by_season(sorted_seasonal_stats)
        # if not user_list_sent:
        #     sorted_seasonal_stats_no_sequels = SeasonalStats.get_user_seasonal_stats(username, site, user_list)
        #     return sorted_seasonal_stats, sorted_seasonal_stats_no_sequels
        # else:
        #     return sorted_seasonal_stats


# stats = SeasonalStats("michhoffman")
# print(5)
