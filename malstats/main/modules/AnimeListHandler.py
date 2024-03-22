from main.modules.AnimeDB import AnimeDB
from main.modules.MAL_utils import MALUtils, get_data
import requests
from abc import ABC, abstractmethod
import polars as pl
from main.modules.Errors import UserDoesNotExistError
from main.modules.general_utils import list_to_uint8_array, timeit, redis_cache_wrapper
from dataclasses import dataclass, field
from main.modules.AnimeList import MALList, AniList
from django.core.cache import cache
# from main.modules.AnimeListFormatter import ListFormatter
# Redis-wrap the fetcher


class AnimeListHandler(ABC):

    def __init__(self, user_name=None, anime_list=[]):
        if not user_name and not anime_list:
            raise AttributeError("Either user_name or anime_list must be initialized")
        self.user_name = user_name
        self._anime_list = anime_list
        self._all_titles = None

    @property
    def anime_list(self):
        pass

    @property
    def all_titles(self):
        if not self._all_titles:
            self._all_titles = self._get_all_list_titles()
        return self._all_titles

    @abstractmethod
    def _fetch_user_anime_list(self):
        """Retrieves the user's anime list from the relevant service."""
        pass

    @abstractmethod
    def get_user_scores_list(self, db_row=False):
        """Get user's scores as a simple list :
        [score_1, score_2, ..., score_n], where score_1
        corresponds to the user's score for the first entry
        in AnimeDB and so on.

        db_row is for internal functionality and should not be used."""
        pass

    # @abstractmethod
    # def _get_all_list_titles(self):
    #     pass

    @staticmethod
    @abstractmethod
    def get_title_from_entry(entry):
        pass

    @staticmethod
    @abstractmethod
    def get_score_from_entry(entry):
        pass

    @staticmethod
    @abstractmethod
    def get_list_status_from_entry(entry):
        pass

    @staticmethod
    @abstractmethod
    def get_num_watched_from_entry(entry):
        pass

    @staticmethod
    def determine_list_site(anime_list):
        if type(anime_list) == MALList:
            return "MAL"
        else:
            return "Anilist"

    @staticmethod
    def get_concrete_handler(type):
        if type == 'MAL':
            return MALListHandler
        elif type == 'Anilist':
            return AnilistHandler
        else:
            raise ValueError("Unknown website, only MAL and Anilist are supported.")

    def get_user_scores_list(self, db_row=False):
        titles = AnimeDB().titles

        anime_indexes = {v: k for (k, v) in enumerate(titles)}
        new_list = [None] * (len(titles))

        show_amount = 0
        score_sum = 0
        for title in self.anime_list:  # anime_list, scores_db
            # title = anime['node']['title']
            score = self.anime_list[title]['score']
            if score == 0:
                continue
            show_amount += 1
            if title in titles:
                new_list[anime_indexes[title]] = score
            score_sum += score

        if not db_row:
            return new_list
        else:
            try:
                mean_score = round(score_sum / show_amount, 4)
            except ZeroDivisionError:
                mean_score = 0  # Users are filtered (must have >50 scored shows), but
                # technically a user can remove all his scores between the time they were added to the db and
                # the time this program runs.
            return new_list, mean_score, show_amount


class MALListHandler(AnimeListHandler):

    def __init__(self, user_name=None, anime_list=[], full_list=True):
        super().__init__(user_name, anime_list)
        self.full_list = full_list

    @property
    def anime_list(self):
        if not self._anime_list:
            self._anime_list = MALList(self._fetch_user_anime_list())
        return self._anime_list

    @timeit
    def _fetch_user_anime_list(self):
        # Manual redis caching, decorator won't work since user_name isn't part of the
        # args here

        cache_key = f"user_list_{self.user_name}_MAL"
        result = cache.get(cache_key)
        if result is not None:
            print(f"Cached result for {self.user_name}'s list found")
            return result

        url = f'https://api.myanimelist.net/v2/users/' \
              f'{self.user_name}/animelist?fields=list_status&limit=1000&sort=list_score&nsfw=True'
        response = get_data(url)
        anime_list = response["data"]

        # If the user has more than 1000 entries in their list, we will need separate API
        # calls for each thousand.

        if len(anime_list) == 1000:
            scored_shows = MALUtils.count_scored_shows(anime_list)
            thousands = 1
            while len(anime_list) == 1000 * thousands and (scored_shows == 1000 * thousands
                                                           or self.full_list):
                print(
                    f'Length of {self.user_name}\'s anime list exceeds {1000 * thousands}, '  #
                    f'proceeding to next 1000 shows')
                url = f'https://api.myanimelist.net/v2/users/' \
                      f'{self.user_name}/animelist?fields=list_status&limit=1000&sort' \
                      f'=list_score' \
                      f'&offset={1000 * thousands}&nsfw=True'  #
                response = get_data(url)
                next_part = response["data"]
                anime_list = anime_list + next_part
                thousands += 1
                scored_shows = MALUtils.count_scored_shows(anime_list)

        cache.set(cache_key, anime_list, 60*5)
        return anime_list

    def get_user_scores_list(self, db_row=False):
        titles = AnimeDB().titles

        anime_indexes = {v: k for (k, v) in enumerate(titles)}
        new_list = [None] * (len(titles))

        show_amount = 0
        score_sum = 0
        for title in self.anime_list:  # anime_list, scores_db
            # title = anime['node']['title']
            score = self.anime_list[title]['score']
            if score == 0:
                break
            show_amount += 1
            if title in titles:
                new_list[anime_indexes[title]] = score
            score_sum += score

        if not db_row:
            return new_list
        else:
            try:
                mean_score = round(score_sum / show_amount, 4)
            except ZeroDivisionError:
                mean_score = 0  # Users are filtered (must have >50 scored shows), but
                # technically a user can remove all his scores between the time they were added to the db and
                # the time this program runs.
            return new_list, mean_score, show_amount

    def _get_all_list_titles(self):
        return [anime['node']['title'] for anime in self.anime_list]

    @staticmethod
    def get_title_from_entry(entry):
        return entry['node']['title']

    @staticmethod
    def get_score_from_entry(entry):
        return entry['list_status']['score']

    @staticmethod
    def get_list_status_from_entry(entry):
        return entry['list_status']['status']

    @staticmethod
    def get_num_watched_from_entry(entry):
        return entry['list_status']['num_episodes_watched']


class AnilistHandler(AnimeListHandler):

    def __init__(self, user_name=None, anime_list=[]):
        super().__init__(user_name, anime_list)

    def __iter__(self):
        # Attempt to iterate over the main value, regardless of its type
        return iter(self.anime_list)

    @property
    def anime_list(self):
        if not self._anime_list:
            self._anime_list = AniList(self._fetch_user_anime_list())
        return self._anime_list

    @timeit
    def _fetch_user_anime_list(self, full_list=True):
        query = '''
        query ($userName: String) {
      MediaListCollection(userName: $userName, type: ANIME) {
        lists {
          name
          entries {
            media {
            
              title {
                userPreferred
              }
              coverImage {
                large
              }
              episodes
              averageScore
              idMal
            }
            score
            status
            updatedAt
          }
        }
      }
    }
        '''

        cache_key = f"user_list_{self.user_name}_Anilist"
        result = cache.get(cache_key)
        if result is not None:
            return result

        url = 'https://graphql.anilist.co'
        variables = {
            'userName': self.user_name
        }
        anime_list = requests.post(url, json={"query": query, "variables": variables}, timeout=30).json()
        if not anime_list:
            raise UserDoesNotExistError(f"User {self.user_name} does not exist.")
        anime_list = [entry for lst in anime_list['data']['MediaListCollection']['lists'] if lst['name'] != 'Planning'
                      for entry in lst['entries']]

        cache.set(cache_key, anime_list, 60 * 5)
        return anime_list

    # def get_user_scores_list(self, db_row=False):
    #     titles = AnimeDB().titles
    #
    #     anime_indexes = {v: k for (k, v) in enumerate(titles)}
    #     new_list = [None] * (len(titles))
    #
    #     show_amount = 0
    #     score_sum = 0
    #     for lst in self.anime_list['lists']:
    #         if lst['name'] == 'Planning':
    #             continue
    #         for entry in lst['entries']:
    #             entry_MAL_ID = entry['media']['idMal']
    #             if not entry_MAL_ID:
    #                 continue
    #
    #             score = entry['score']
    #             if not score:
    #                 continue
    #
    #             title = MALUtils.get_anime_title_by_id(entry_MAL_ID)
    #             show_amount += 1
    #             if title in titles:
    #                 new_list[anime_indexes[title]] = score
    #             score_sum += score
    #
    #     if not db_row:
    #         return new_list
    #
    #     else:
    #         try:
    #             mean_score = round(score_sum / show_amount, 4)
    #         except ZeroDivisionError:
    #             mean_score = 0  # Users are filtered (must have >50 scored shows), but
    #             # technically a user can remove all his scores between the time they were added to the db and
    #             # the time this program runs.
    #         return new_list, mean_score, show_amount

    def _get_all_list_titles(self):
        all_titles = []
        for lst in self.anime_list['lists']:
            if lst['name'] == 'Planning':
                continue
            for entry in lst['entries']:
                try:
                    mal_title = self.get_title_from_entry(entry)
                except ValueError:
                    continue  # This shouldn't happen since every Anilist entry should
                    # have a corresponding MAL entry, but just in case there are discrepancies.
                all_titles.append(mal_title)
        return all_titles

    @staticmethod
    def get_title_from_entry(entry):
        entry_mal_id = entry['media']['idMal']
        mal_title = MALUtils.get_anime_title_by_id(entry_mal_id)
        return mal_title

    @staticmethod
    def get_score_from_entry(entry):
        return entry['score']

    @staticmethod
    def get_list_status_from_entry(entry):
        return entry['status']

    @staticmethod
    def get_num_watched_from_entry(entry):
        return entry['media']['episodes']
        # Anilist currently doesn't have num watched info, this returns
        # episode number





