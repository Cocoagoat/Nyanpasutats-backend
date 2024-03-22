import numpy as np
from main.modules.AnimeDB import AnimeDB
from main.modules.AnimeListFormatter import ListFormatter
from main.modules.AnimeListHandler import AnimeListHandler
from main.modules.MAL_utils import *
from main.models import AnimeData
from main.modules.Tags import Tags
from collections import defaultdict
from scipy import stats as scipy_stats

from main.modules.general_utils import timeit


class SeasonalStats:
    # I will fix this mess at some point, it was written in a very frantic day
    def __init__(self):
        pass

    @staticmethod
    @timeit
    def get_user_seasonal_stats(username, site, user_list=None, min_shows=5):

        MIN_SHOWS_IN_SEASON = 5

        def sort_by_season(stats_dict):
            def get_season_value(season_data):
                season, year = season_data[0].split(" ")
                season_value = (Seasons[season].value)/10
                return int(year) + season_value

            sorted_stats = {season: season_stats for season, season_stats in sorted(stats_dict.items(),
                                                                           key=lambda x: get_season_value(x),
                                                                           reverse=True) if season_stats['Shows'] >= 5}
            return sorted_stats

        def add_to_existing_season(seasonal_dict, season, user_list_status, user_score, anime):
            if user_list_status == 'dropped':
                seasonal_dict[season]['DroppedShows'] += 1
            if user_score:
                seasonal_dict[season]['Score'] += user_score
                seasonal_dict[season]['ScoredShows'] += 1
            seasonal_dict[season]['Shows'] += 1
            seasonal_dict[season]['ShowList'] = seasonal_dict[season]['ShowList'] | {anime.name: user_score}

        def create_and_add_to_new_season(seasonal_dict, anime, user_score):
            seasonal_dict[season] = {'Score': user_score, 'Shows': 1, 'ScoredShows': 1,
                                     'ShowList': {anime.name: user_score}, 'DroppedShows': 0}

        def sort_season_shows_by_scores(shows):
            return {show: score for show, score in sorted(
                shows.items(), reverse=True, key=lambda x: x[1])}

        def filter_seasonal_dict_by_show_amount(seasonal_dict, min_show_amount):
            return {season: season_stats for season, season_stats in seasonal_dict.items()
                    if season_stats['Shows'] >= min_show_amount}

        def calculate_favs_avg_score(seasonal_dict, season):
            # avg_score = seasonal_dict[season]['Score'] / seasonal_dict[season]['ScoredShows']

            if seasonal_dict[season]['Shows'] >= 10:
                scores_of_favorites = list(seasonal_dict[season]['ShowList'].values())[0:10]
                sum_of_favorites = sum((score for score in scores_of_favorites))
                fav_avg_score = sum_of_favorites / 10
            else:
                fav_avg_score = seasonal_dict[season]['AvgScore']

            del seasonal_dict[season]['Score']
            del seasonal_dict[season]['ScoredShows']
            return fav_avg_score

        def find_most_controversial_score_in_season(season_stats, no_controversial_found=False):
            max_diff = 0
            MAL_scores = {anime.name: float(anime.mean_score) for anime in anime_data.filter(
                name__in=season_stats['ShowList'].keys())}

            for show, score in season_stats['ShowList'].items():
                if score == 0:
                    continue
                MAL_score = MAL_scores[show]
                diff = abs(score - MAL_score)
                controversial_conditions = [MAL_score < 6 and score >= 7,
                                            MAL_score < 7 and score >= 8,
                                            MAL_score < 7.5 and score >= 9,
                                            MAL_score < 8 and score == 10,
                                            MAL_score > 9 and score <= 7,
                                            MAL_score > 8 and score <= 6,
                                            MAL_score > 7.5 and score <= 5,
                                            MAL_score > 7 and score <= 4]
                controversial = any(controversial_conditions)
                if diff > max_diff and (controversial or no_controversial_found):
                    max_diff = diff
                    max_diff_show = show

            if max_diff == 0:
                max_diff_show = find_most_controversial_score_in_season(season_stats, no_controversial_found=True)
            return max_diff_show

        def get_season_rank(sorted_seasonal_stats, prev_season, season, i, favorites=False):
            if not prev_season:
                return 1
            avg_type = 'AvgScore' if not favorites else 'FavoritesAvgScore'
            rank_type = 'OverallRank' if not favorites else 'FavoritesRank'
            if sorted_seasonal_stats[prev_season][avg_type] == sorted_seasonal_stats[season][avg_type]:
                rank = sorted_seasonal_stats[prev_season][rank_type]
            else:
                rank = i
            return rank

        def get_most_watched_genre(show_list):
            genre_counts = defaultdict(int)
            for show in show_list:
                try:
                    show_genres = tags.entry_tags_dict[show]['Genres']
                except KeyError:
                    continue
                for genre in show_genres:
                    genre_counts[genre] += 1
            try:
                return max(genre_counts, key=genre_counts.get)
            except ValueError:
                return "None"

        def calculate_contrarian_index(user_show_list, user_season_show_list):
            user_scores = np.array([user_show_list[title]['score'] for title in user_season_show_list])
            MAL_scores = np.array([float(AnimeData.objects.get(name=title).mean_score) for title in user_season_show_list])
            non_zero_mask = (user_scores != 0) & (MAL_scores != 0)

            user_scores = user_scores[non_zero_mask]
            MAL_scores = MAL_scores[non_zero_mask]
            affinity = np.corrcoef(user_scores, MAL_scores)[0][1]
            return affinity if not np.isnan(affinity) else 0.5

        def get_total_shows_duration(show_list, user_titles):
            show_obj_list = [AnimeData.objects.get(name=title) for title in show_list]
            return sum([show.duration * user_titles[show.name]['num_watched'] for show in show_obj_list])

        tags = Tags()
        user_list_sent = bool(user_list)
        if not user_list: # send list_handler instead of user_list
            ListHandler = AnimeListHandler.get_concrete_handler(site)
            user_list = ListHandler(username).anime_list.list
            user_list = {show: stats for show, stats in user_list.items()
                         if stats['list_status'] == "dropped" or stats['score']}
        else:
            # Filtering out sequels
            user_list = {title: stats for title, stats in user_list.items() if title in tags.show_tags_dict.keys()}

        anime_data = AnimeData.objects.filter(name__in=user_list.keys())  # The SQLite DB object
        seasonal_dict = {}
        for anime in anime_data:

            if anime.episodes >= 6:  # OVAs or movies don't count as part of the season
                try:
                    season_name = Seasons(anime.season).name  # Seasons are listed as 1,2,3,4 in the database
                except AttributeError:
                    continue

                user_score = user_list[anime.name]['score']
                user_list_status = user_list[anime.name]['list_status']

                season = f"{season_name} {anime.year}"
                if not season:
                    continue

                if season in seasonal_dict.keys():
                    add_to_existing_season(seasonal_dict, season, user_list_status, user_score, anime)
                else:
                    create_and_add_to_new_season(seasonal_dict, anime, user_score)

        seasonal_dict = filter_seasonal_dict_by_show_amount(seasonal_dict, min_show_amount=MIN_SHOWS_IN_SEASON)

        for season in seasonal_dict.keys():
            seasonal_dict[season]['AvgScore'] = seasonal_dict[season]['Score'] / seasonal_dict[season]['ScoredShows']

        sorted_seasonal_stats = {season: stats for season, stats in sorted(seasonal_dict.items(),
                                                                 key=lambda x: x[1]['AvgScore'],
                                                                 reverse=True) if stats['ScoredShows'] >= min_shows}


        prev_season = None
        for i, (season, season_stats) in enumerate(sorted_seasonal_stats.items(), start=1):

            sorted_seasonal_stats[season]['OverallRank'] = get_season_rank(sorted_seasonal_stats,
                                                                           prev_season, season, i)

            sorted_seasonal_stats[season]['MostWatchedGenre'] = get_most_watched_genre(season_stats['ShowList'])
            sorted_seasonal_stats[season]['MostControversialShow'] = find_most_controversial_score_in_season(season_stats)
            sorted_seasonal_stats[season]['Affinity'] = calculate_contrarian_index(user_list, season_stats['ShowList'])
            sorted_seasonal_stats[season]['ShowList'] = sort_season_shows_by_scores(seasonal_dict[season]['ShowList'])
            sorted_seasonal_stats[season]['FavoritesAvgScore'] = calculate_favs_avg_score(
                sorted_seasonal_stats, season)
            sorted_seasonal_stats[season]['TotalShowsDuration'] = get_total_shows_duration(seasonal_dict[season]['ShowList'],
                                                                                           user_list)
            prev_season = season


        sorted_by_favs_seasonal_stats = {season: stats for season, stats in sorted(sorted_seasonal_stats.items(),
                                                                           key=lambda x: x[1]['FavoritesAvgScore'],
                                                                           reverse=True)}

        prev_season = None
        for i, (season, _) in enumerate(sorted_by_favs_seasonal_stats.items(), start=1):
            sorted_seasonal_stats[season]['FavoritesRank'] = get_season_rank(sorted_by_favs_seasonal_stats,
                                                                             prev_season, season, i, favorites=True)
            prev_season = season

        current_year = datetime.date.today().year

        years_counter = {str(year): 0 for year in range(1960, current_year+1)}

        for season, season_stats in sorted_seasonal_stats.items():
            season_name, year = season.split(" ")
            season_stats['YearlyRank'] = years_counter[year] + 1
            years_counter[year] += 1

        years_counter = {str(year): 0 for year in range(1960, current_year+1)}

        for season, season_stats in sorted_by_favs_seasonal_stats.items():
            season_name, year = season.split(" ")
            season_stats['FavYearlyRank'] = years_counter[year] + 1
            years_counter[year] += 1

        sorted_seasonal_stats = sort_by_season(sorted_seasonal_stats)
        if not user_list_sent:
            sorted_seasonal_stats_no_sequels = SeasonalStats.get_user_seasonal_stats(username, site, user_list)
            return sorted_seasonal_stats, sorted_seasonal_stats_no_sequels
        else:
            return sorted_seasonal_stats


# stats = SeasonalStats("michhoffman")
# print(5)
