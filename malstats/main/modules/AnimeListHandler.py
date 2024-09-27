from main.modules.AnimeDB import AnimeDB
from main.modules.MAL_utils import MALUtils, get_data
import requests
import logging
from abc import ABC, abstractmethod
from main.modules.filenames import anime_database_updated_name, anime_database_name
from main.modules.Errors import UserDoesNotExistError, UserListFetchError
from main.modules.AnimeList import MALList, AniList
from main.modules.general_utils import rate_limit
from django.core.cache import cache

logger = logging.getLogger("nyanpasutats")


class AnimeListHandler(ABC):
    """Usage to get user's list :
    anime_list = MALListHandler(user_name).anime_list will return the AnimeList object,
    which has both the raw list as returned from MAL and the simplified  formatted version.
    Use anime_list.list_obj to access the former and anime_list.list to access the latter."""

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

    @staticmethod
    def get_concrete_handler(type):
        if type == 'MAL':
            return MALListHandler
        elif type == 'Anilist':
            return AnilistHandler
        else:
            raise ValueError("Unknown website, only MAL and Anilist are supported.")

    def get_user_scores_list(self, db_row=False, for_predict=False):
        """Get user's scores as a simple list :
         [score_1, score_2, ..., score_n], where score_1
         corresponds to the user's score for the first entry
         in AnimeDB and so on.

         db_row is for internal functionality and should not be used."""
        titles = AnimeDB(anime_database_updated_name if for_predict else anime_database_name).titles

        anime_indexes = {v: k for (k, v) in enumerate(titles)}
        new_list = [None] * (len(titles))

        show_amount = 0
        score_sum = 0
        for title in self.anime_list:  # anime_list, scores_db
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
    """Usage to get user's list :

       anime_list = MALListHandler(user_name).anime_list will return the AnimeList object,
       which has both the raw list as returned from MAL and the simplified  formatted version.

       Use anime_list.list_obj to access the former and anime_list.list to access the latter."""

    def __init__(self, user_name=None, anime_list=[], full_list=True):
        super().__init__(user_name, anime_list)
        self.full_list = full_list

    @property
    def anime_list(self):
        if not self._anime_list:
            self._anime_list = MALList(self._fetch_user_anime_list())
        return self._anime_list

    @rate_limit(rate_lim=5, cache_key="MAL_rate_limit")
    def _send_request_for_list(self, url):
        """This function exists purely to be decorated with @rate_limit, the inner
        get_data function is more generic and cannot receive this MAL-specific decorator"""
        return get_data(url)

    def _fetch_user_anime_list(self):
        # Manual redis caching, decorator won't work since user_name isn't part of the
        # args here

        cache_key = f"user_list_{self.user_name}_MAL"
        result = cache.get(cache_key)
        if result is not None:
            logger.info(f"Cached result for {self.user_name}'s list found")
            return result

        url = f'https://api.myanimelist.net/v2/users/' \
              f'{self.user_name}/animelist?fields=list_status&limit=1000&sort=list_score&nsfw=True'
        logger.info(f"Fetching list for user {self.user_name}")
        response = self._send_request_for_list(url)
        anime_list = response["data"]

        # If the user has more than 1000 entries in their list, we will need separate API
        # calls for each thousand.

        if len(anime_list) == 1000:
            scored_shows = MALUtils.count_scored_shows(MALList(anime_list))
            thousands = 1
            while len(anime_list) == 1000 * thousands and (scored_shows == 1000 * thousands
                                                           or self.full_list):
                logger.info(
                    f'Length of {self.user_name}\'s anime list exceeds {1000 * thousands}, '  #
                    f'proceeding to next 1000 shows')
                url = f'https://api.myanimelist.net/v2/users/' \
                      f'{self.user_name}/animelist?fields=list_status&limit=1000&sort' \
                      f'=list_score' \
                      f'&offset={1000 * thousands}&nsfw=True'
                response = self._send_request_for_list(url)
                next_part = response["data"]
                anime_list = anime_list + next_part
                thousands += 1
                scored_shows = MALUtils.count_scored_shows(MALList(anime_list))

        cache.set(cache_key, anime_list, timeout=900)
        logger.info(f"Successfully fetched list for user {self.user_name}")
        return anime_list

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
              User(name: $userName) {
                mediaListOptions {
                  scoreFormat
                }
              }
            }
                '''

    url = 'https://graphql.anilist.co'

    def __init__(self, user_name=None, anime_list=[]):
        super().__init__(user_name, anime_list)

    def __iter__(self):
        return iter(self.anime_list)

    @property
    def anime_list(self):
        if not self._anime_list:
            self._anime_list = AniList(self._fetch_user_anime_list())
        return self._anime_list

    @rate_limit(rate_lim=5, cache_key="Anilist_rate_limit")
    def _send_request_for_list(self, variables):
        return requests.post(self.url, json={"query": self.query,
                                             "variables": variables}, timeout=30).json()

    def _adjust_list_for_score_system(self, anime_list, score_system):
        if score_system == 100:
            for anime in anime_list:
                anime['score'] /= 10
        elif score_system == 5:
            query = '''
                        query ($userName: String) {
                      User(name: $userName) {
                        mediaListOptions {
                          scoreFormat
                        }
                      }
                    }
                        '''
            variables = {
                'userName': self.user_name
            }
            score_style = requests.post(self.url, json={"query": query,
                                          "variables": variables}, timeout=30).json()
            score_style = score_style['data']['User']['mediaListOptions']['scoreFormat']
            if score_style == "POINT_5":
                for anime in anime_list:
                    anime['score'] *= 2
            elif score_style == "POINT_3":
                # Actual psychopaths
                for anime in anime_list:
                    anime['score'] *= (10/3)
        return anime_list

    def _fetch_user_anime_list(self, full_list=True):

        cache_key = f"user_list_{self.user_name}_Anilist"
        result = cache.get(cache_key)
        if result is not None:
            return result

        variables = {
            'userName': self.user_name
        }
        anime_list = self._send_request_for_list(variables)
        if not anime_list:
            raise UserDoesNotExistError(f"This user does not exist.")

        try:
            anime_list = [entry for lst in anime_list['data']['MediaListCollection']['lists']
                          if lst['name'] != 'Planning'
                          for entry in lst['entries']]
        except TypeError:
            if 'errors' in anime_list.keys():
                if anime_list['errors'][0]['status'] == 404:
                    raise UserListFetchError("This user's list is private.")

        if not anime_list:
            raise UserListFetchError(f"This user's list has no scored shows.")

        anime_list = sorted(anime_list, reverse=True, key=lambda x: x['score'])
        if anime_list[0]['score'] > 10:
            self._adjust_list_for_score_system(anime_list, score_system=100)
        elif anime_list[0]['score'] <= 5:
            self._adjust_list_for_score_system(anime_list, score_system=5)

        cache.set(cache_key, anime_list, timeout=900)
        return anime_list

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





