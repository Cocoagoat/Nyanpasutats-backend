from main.modules.AnimeDB import AnimeDB
from main.modules.MAL_utils import *


class SeasonalStats:

    def __init__(self,start_year=1900, end_year=2100, min_shows=5, non_sequels_only=False):
        # self.user_name = user_name
        self._stats = None
        self.start_year = start_year
        self.end_year = end_year
        self.min_shows = min_shows
        self.non_sequels_only = non_sequels_only

    @property
    def stats(self):
        if not self._stats:
            self._stats = self.get_user_seasonal_stats()
        return self._stats

    def filter(self):
        filtered_dict = {season: season_stats for season, season_stats in self.stats.items()
                         if int(season.split(" ")[1]) >= self.start_year and int(season.split(" ")[1]) <= self.end_year
                         and season_stats['Shows'] >= self.min_shows}
        return filtered_dict

    # def filter_by_threshold(self, min_shows):
    #     filtered_dict = {season : season_stats for season, season_stats in self.avg_score_per_season.items()
    #                      if season_stats['Shows'] >= min_shows}
    #     return filtered_dict

    @staticmethod
    def get_user_seasonal_stats(username):

        anime_db = AnimeDB()
        user_list = get_user_MAL_list(username)
        user_titles = {anime['node']['title']: anime['list_status']['score']
                       for anime in user_list if anime['list_status']['score']}
        # show_seasons = anime_db.df[anime_db.stats['Season']]
        show_seasons = anime_db.seasons
        show_years = anime_db.years
        # show_years = anime_db.df[anime_db.stats['Year']]
        # show_types = anime_db.df[anime_db.stats['Type']]
        show_episodes = anime_db.episodes

        seasonal_dict = {}
        for show, score in user_titles.items():
            try:
                show_year = int(show_years[show])
                show_season = Seasons(show_seasons[show]).name
            except KeyError:
                continue

            season = f"{show_season} {show_year}"
            if show_episodes[show] >= 6:
                if season in seasonal_dict.keys():
                    seasonal_dict[season]['Score'] += score
                    seasonal_dict[season]['Shows'] += 1
                    seasonal_dict[season]['ShowList'] = seasonal_dict[season]['ShowList'] | {show : score}
                else:
                    seasonal_dict[season] = {'Score': score, 'Shows': 1, 'ShowList': {show: score}}

        for season in seasonal_dict.keys():
            # seasonal_dict[season]['AboveThreshold'] = seasonal_dict[season]['Shows'] >= 5
            seasonal_dict[season]['AvgScore'] = seasonal_dict[season]['Score'] / seasonal_dict[season]['Shows']
            del seasonal_dict[season]['Score']

        sorted_dict = {season: stats for season, stats in sorted(seasonal_dict.items(),
                                                                 key=lambda x: x[1]['AvgScore'],
                                                                 reverse=True)}

        return sorted_dict
            # seasonal_dict = {season : seasonal_dict[season]['Score']/seasonal_dict[season]['Shows']
            #                      for season in seasonal_dict.keys() if seasonal_dict[season]['Shows'] >= 5}
            # seasonal_dict = sorted()


stats = SeasonalStats("BaronBrixius")
print(5)
