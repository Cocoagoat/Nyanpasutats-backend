from __future__ import print_function
from main.modules.general_utils import get_data, Sleep
from bs4 import BeautifulSoup
import datetime
import time
from enum import Enum
import django
django.setup()
from main.models import AnimeData
try:
    import thread
except ImportError:
    import _thread as thread


class Seasons(Enum):
    Winter = 1
    Spring = 2
    Summer = 3
    Fall = 4


class MediaTypes(Enum):
    tv = 1
    movie = 2
    ova = 3
    special = 4
    ona = 5


class MALUtils:

    status_convert_dict = {'COMPLETED' : 'completed', 'DROPPED' : 'dropped',
                   'CURRENT' : 'watching', 'PAUSED' : 'on_hold',
                   'PLANNING' : 'plan_to_watch', 'REPEATING' : 'completed'}  # No MAL equivalent for repeating

    @staticmethod
    def get_anime_title_by_id(mal_id):
        try:
            show_data = AnimeData.objects.get(mal_id=mal_id)
            return show_data.name
        except AnimeData.DoesNotExist:
            raise ValueError("No anime found with that MAL ID")

    @staticmethod
    def count_scored_shows(user_list):
        count = 0
        for anime in user_list:
            count += 1
            if anime['list_status']['score'] == 0:
                return count
        return count

    @staticmethod
    def get_anime_batch_from_MAL(page_num, required_fields):

        url = f'https://api.myanimelist.net/v2/anime/ranking?ranking_type=all&limit=100&offset={page_num * 100}&' \
              'fields='
        for field in required_fields:
            url = url + field + ","  # Concatenate all required fields with a comma inbetween
        url = url[:-1]  # Remove the last comma
        anime_batch = get_data(url)
        # anime_batch = call_function_through_process(get_search_results, url)
        while anime_batch is None:
            # We keep retrying until we get the batch, the URL is correct so the main reason
            # for failure would be the API itself being down, in which case we just wait for it
            # to come back up.
            print("Anime batch was returned as None, sleeping and retrying")
            # logger.debug("Anime batch was returned as None, sleeping and retrying")
            time.sleep(Sleep.MEDIUM_SLEEP)
            # anime_batch = call_function_through_process(get_search_results, url)
            anime_batch = get_data(url)
        return anime_batch

    @staticmethod
    def check_account_age_directly(user_name):
        """ Used in case we couldn't verify that the user's account is older than 30 days
        through their anime list (could be an Anilist migrator). In this case, we have to make
        an extra API call to make sure the user wasn't created in the last 30 days."""
        url = f'https://myanimelist.net/profile/{user_name}'
        page_html = get_data(url)
        if not page_html:
            page_html = get_data(url)
            if not page_html:
                return datetime.datetime.now(datetime.timezone.utc)
                # Something wrong with the page html, very rare

        soup = BeautifulSoup(page_html.text, "html.parser")
        date_str = soup.findAll("span", {"class": "user-status-data di-ib fl-r"})[
            -1].text  # We'll always need the last one
        date_obj = datetime.datetime.strptime(date_str, "%b %d, %Y")
        account_creation_time = date_obj.replace(tzinfo=datetime.timezone.utc)
        current_time = datetime.datetime.now(datetime.timezone.utc)
        time_since_account_creation = current_time - account_creation_time
        return time_since_account_creation

    @staticmethod
    def get_usernames_from_show(base_url):
        """This function takes an anime title, accesses the updates table on the Stats
        tab on the MAL page of that title, and returns pages_to_get*75 usernames from it.
        This is the easiest way to gather recently active usernames."""
        pages_to_get = 10
        users_table = []
        too_many_requests_flag = 0
        page_num = 0
        while page_num < pages_to_get:
            page_users_table = []
            url = f"{base_url}show={str(75 * page_num)}"
            print(f"Getting usernames from {url}")
            # logger.debug(f"Getting usernames from {url}")
            page_html = get_data(url)
            try:
                soup = BeautifulSoup(page_html.text, "html.parser")
                update_table = soup.find("table", {"class": "table-recently-updated"})
                page_users_table = update_table.findAll('tr')[1:]
                too_many_requests_flag = 0
            # flag gets reset to 0 if succeeded in getting data from soup
            except AttributeError:
                if too_many_requests_flag:
                    # flag being 1 means we got an error twice in a row,
                    # most likely an issue with too many requests
                    print("Too many requests, commencing sleep then trying again")
                    # logger.debug("Too many requests, commencing sleep then trying again")
                    time.sleep(Sleep.LONG_SLEEP)
                else:
                    base_url = f"{base_url}m=all&"
                    page_num = page_num - 1
                too_many_requests_flag = 1
            users_table = users_table + page_users_table
            page_num = page_num + 1
        print(f'Length of the table returned from get_usernames_from_show'
              f' : {len(users_table)}')  #
        return users_table

    @staticmethod
    def replace_characters_for_url(s):
        """This function replaces characters in the titles of the shows as they are
        returned by the MAL API,
        according to the way they're supposed to be written in the URL. There doesn't
        seem to be any consistency -
        some characters like "-" are retained as they are, while others are omitted or
        replaced with an underscore.
        Seriously, exclamation marks are omitted while question marks sometimes
        are...and sometimes aren't."""
        # This was written years ago, replace with encodeCharacterURI equivalent later

        replace_with_empty = [",", "(", ")", "'", ".", "!", "\"", "?", "[", "]", "{", "}",
                              "$", "#", "@", "%", "^", "*"]  #
        replace_with_underscore = [" ", ":", "&", ";", "/", "+"]
        for ch in replace_with_empty:
            s = s.replace(ch, "")
        for ch in replace_with_underscore:
            s = s.replace(ch, "_")
        return s

    @classmethod
    #  move to AnilistUtils?
    def convert_anilist_status_to_MAL(cls, anilist_status):
        return cls.status_convert_dict[anilist_status]



# @timeit
# # @redis_cache_wrapper(timeout=5*60)
# def get_user_anime_list(user_name, anilist=False, full_list=True):
#
#     # cache_key = f'user_list_{user_name}'
#     # cached_result = cache.get(cache_key)
#     # if cached_result:
#     #     return cached_result
#
#     if anilist:
#         user_list = get_user_anilist(user_name, full_list)
#     else:
#         user_list = get_user_MAL_list(user_name, full_list)
#
#     # cache.set(cache_key, anime_list, CACHE_TIMEOUT)
#     return user_list





