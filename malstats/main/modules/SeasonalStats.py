from main.modules.AnimeDB import AnimeDB
from main.modules.MAL_utils import *
from main.models import AnimeData


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
    def get_user_seasonal_stats2(username):

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
                    seasonal_dict[season]['ShowList'] = seasonal_dict[season]['ShowList'] | {show: score}
                else:
                    seasonal_dict[season] = {'Score': score, 'Shows': 1, 'ShowList': {show: score}}

        for season in seasonal_dict.keys():
            # seasonal_dict[season]['AboveThreshold'] = seasonal_dict[season]['Shows'] >= 5
            seasonal_dict[season]['AvgScore'] = seasonal_dict[season]['Score'] / seasonal_dict[season]['Shows']

            # top_half_amount = int(np.ceil(seasonal_dict[season]['Shows']/2))

            if seasonal_dict[season]['Shows'] >= 10:
                scores_of_favorites = list(seasonal_dict[season]['ShowList'].values())[0:10]
                sum_of_favorites = sum([score for score in scores_of_favorites])
                seasonal_dict[season]['FavoritesAvgScore'] = sum_of_favorites/10
            else:
                seasonal_dict[season]['FavoritesAvgScore'] = seasonal_dict[season]['AvgScore']
            del seasonal_dict[season]['Score']

        sorted_dict = {season: stats for season, stats in sorted(seasonal_dict.items(),
                                                                 key=lambda x: x[1]['AvgScore'],
                                                                 reverse=True) if stats['Shows'] >= 5}

        sorted_dict2 = {season: stats for season, stats in sorted(seasonal_dict.items(),
                                                                 key=lambda x: x[1]['FavoritesAvgScore'],
                                                                 reverse=True) if stats['Shows'] >= 5}

        for season, season_stats in sorted_dict2.items():
            max_diff = 0
            MAL_scores = {anime.name: float(anime.mean_score) for anime in AnimeData.objects.filter(
                name__in=season_stats['ShowList'].keys())}

            for show, score in season_stats['ShowList'].items():
                MAL_score = MAL_scores[show]
                diff = abs(score - MAL_score)
                controversial = MAL_score < 7 and score > MAL_score or MAL_score > 7 and score < MAL_score
                if diff > max_diff and controversial:
                    max_diff = diff
                    max_diff_show = show
            sorted_dict2[season]['MostControversialShow'] = max_diff_show

        return sorted_dict2
            # seasonal_dict = {season : seasonal_dict[season]['Score']/seasonal_dict[season]['Shows']
            #                      for season in seasonal_dict.keys() if seasonal_dict[season]['Shows'] >= 5}
            # seasonal_dict = sorted()

    @staticmethod
    def get_user_seasonal_stats(username):

        user_list = get_user_MAL_list(username)
        user_titles = {anime['node']['title']: {'score' :anime['list_status']['score'],
                                                'status' : anime['list_status']['status']}
                       for anime in user_list if anime['list_status']['score']
                       or anime['list_status']['status'] == 'dropped'}

        anime_data = AnimeData.objects.filter(name__in=user_titles.keys())
        # show_seasons = anime_db.df[anime_db.stats['Season']]
        # show_seasons = anime_db.seasons
        # show_years = anime_db.years
        # # show_years = anime_db.df[anime_db.stats['Year']]
        # # show_types = anime_db.df[anime_db.stats['Type']]
        # show_episodes = anime_db.episodes

        seasonal_dict = {}
        for anime in anime_data:

            if anime.episodes >= 6:
                try:
                    season_name = Seasons(anime.season).name
                except AttributeError:
                    continue

                user_score = user_titles[anime.name]['score']
                user_list_status = user_titles[anime.name]['status']

                season = f"{season_name} {anime.year}"

                if season in seasonal_dict.keys():
                    if user_list_status == 'dropped':
                        seasonal_dict[season]['DroppedShows'] += 1
                    if user_score:
                        seasonal_dict[season]['Score'] += user_score
                        seasonal_dict[season]['ScoredShows'] += 1
                    seasonal_dict[season]['Shows'] += 1
                    seasonal_dict[season]['ShowList'] = seasonal_dict[season]['ShowList'] | {anime.name: user_score}
                else:
                    seasonal_dict[season] = {'Score': user_score, 'Shows': 1, 'ScoredShows': 1,
                                             'ShowList': {anime.name: user_score}, 'DroppedShows': 0}

            seasonal_dict[season]['ShowList'] = {show: score for show, score in sorted(
                seasonal_dict[season]['ShowList'].items(), reverse=True, key=lambda x: x[1])}

        seasonal_dict = {season: season_stats for season, season_stats in seasonal_dict.items()
                         if season_stats['Shows'] >= 5}

        for season in seasonal_dict.keys():
            # seasonal_dict[season]['AboveThreshold'] = seasonal_dict[season]['Shows'] >= 5
            seasonal_dict[season]['AvgScore'] = seasonal_dict[season]['Score'] / seasonal_dict[season]['ScoredShows']

            # top_half_amount = int(np.ceil(seasonal_dict[season]['Shows']/2))

            if seasonal_dict[season]['Shows'] >= 10:
                scores_of_favorites = list(seasonal_dict[season]['ShowList'].values())[0:10]
                sum_of_favorites = sum((score for score in scores_of_favorites))
                seasonal_dict[season]['FavoritesAvgScore'] = sum_of_favorites/10
            else:
                seasonal_dict[season]['FavoritesAvgScore'] = seasonal_dict[season]['AvgScore']

            del seasonal_dict[season]['Score']
            del seasonal_dict[season]['ScoredShows']

        sorted_seasonal_stats = {season: stats for season, stats in sorted(seasonal_dict.items(),
                                                                 key=lambda x: x[1]['AvgScore'],
                                                                 reverse=True) if stats['Shows'] >= 5}

        # sorted_dict2 = {season: stats for season, stats in sorted(seasonal_dict.items(),
        #                                                          key=lambda x: x[1]['FavoritesAvgScore'],
        #                                                          reverse=True) if stats['Shows'] >= 5}

        for i, (season, season_stats) in enumerate(sorted_seasonal_stats.items(), start=1):
            max_diff = 0
            sorted_seasonal_stats[season]['OverallRank'] = i
            MAL_scores = {anime.name: float(anime.mean_score) for anime in anime_data.filter(
                name__in=season_stats['ShowList'].keys())}

            for show, score in season_stats['ShowList'].items():
                MAL_score = MAL_scores[show]
                diff = abs(score - MAL_score)
                controversial = MAL_score < 7 and score > MAL_score or MAL_score > 7 and score < MAL_score
                if diff > max_diff and controversial:
                    max_diff = diff
                    max_diff_show = show
            sorted_seasonal_stats[season]['MostControversialShow'] = max_diff_show

        favorites_ranking = sorted(seasonal_dict.items(), key=lambda x: x[1]['FavoritesAvgScore'], reverse=True)

        for i, (season, _) in enumerate(favorites_ranking, start=1):
            sorted_seasonal_stats[season]['FavoritesRank'] = i

        current_year = datetime.date.today().year
        years_dict = {str(year): {} for year in range(1960, current_year+1)}

        for season, season_stats in sorted_seasonal_stats.items():
            season_name, year = season.split(" ")
            season_stats['YearlyRank'] = len(years_dict[year]) + 1
            years_dict[year][season_name] = season_stats

        years_counter = {str(year): 0 for year in range(1960, current_year+1)}

        for season, season_stats in favorites_ranking:
            season_name, year = season.split(" ")
            sorted_seasonal_stats[season]['FavYearlyRank'] = years_counter[year] + 1
            years_dict[year][season_name]['FavYearlyRank'] = years_counter[year] + 1
            years_counter[year] += 1

        return sorted_seasonal_stats

stats = SeasonalStats("BaronBrixius")
print(5)
