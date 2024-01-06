from __future__ import print_function

from polars import ColumnNotFoundError

from main.modules.general_utils import *
from bs4 import BeautifulSoup
import datetime
import time
import pandas as pd
from enum import Enum
import logging
import csv
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from random import shuffle
try:
    import thread
except ImportError:
    import _thread as thread
from operator import itemgetter
from colorama import Fore
# from sortedcollections import OrderedSet
import random
from main.modules.filenames import *
from main.modules.Errors import UserListFetchError, UserListPrivateError


class Seasons(Enum):
    # Saved for future use since we store seasons as numbers rather than strings
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


# class UserListFetchError(Exception):
#     def __init__(self, message, username, default_message):
#         super().__init__(message if message else default_message)
#         self.username = username
#         self.message = self.args[0]
#
#
# class UserDoesNotExistError(UserListFetchError):
#     def __init__(self, message, username):
#         super().__init__(message, username, "User does not exist.")
#
#
# class UserListPrivateError(UserListFetchError):
#     def __init__(self, message, username):
#         super().__init__(message, username, "User list is private.")


def replace_characters_for_url(s):
    """This function replaces characters in the titles of the shows as they are
    returned by the MAL API,
    according to the way they're supposed to be written in the URL. There doesn't
    seem to be any consistency -
    some characters like "-" are retained as they are, while others are omitted or
    replaced with an underscore.
    Seriously, exclamation marks are omitted while question marks sometimes
    are...and sometimes aren't."""

    replace_with_empty = [",", "(", ")", "'", ".", "!", "\"", "?", "[", "]", "{", "}",
                          "$", "#", "@", "%", "^", "*"]  #
    replace_with_underscore = [" ", ":", "&", ";", "/", "+"]
    for ch in replace_with_empty:
        s = s.replace(ch, "")
    for ch in replace_with_underscore:
        s = s.replace(ch, "_")
    return s


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
        logger.debug(f"Getting usernames from {url}")
        page_html = get_search_results(url)
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
                logger.debug("Too many requests, commencing sleep then trying again")
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


def check_account_age_directly(user_name):
    """ Used in case we couldn't verify that the user's account is older than 30 days
    through their anime list (could be an Anilist migrator). In this case, we have to make
    an extra API call to make sure the user wasn't created in the last 30 days."""
    url = f'https://myanimelist.net/profile/{user_name}'
    page_html = get_search_results(url)
    if not page_html:
        page_html = get_search_results(url)
        if not page_html:
            return datetime.datetime.now(datetime.timezone.utc)
            # Something wrong with the page html, very rare

    soup = BeautifulSoup(page_html.text, "html.parser")
    date_str = soup.findAll("span", {"class": "user-status-data di-ib fl-r"})[-1].text #We'll always need the last one
    date_obj = datetime.datetime.strptime(date_str, "%b %d, %Y")
    account_creation_time = date_obj.replace(tzinfo=datetime.timezone.utc)
    current_time = datetime.datetime.now(datetime.timezone.utc)
    time_since_account_creation = current_time - account_creation_time
    return time_since_account_creation


def get_anime_batch_from_jikan(page_num):
    """Utility function to get a batch of 25 shows from the Jikan API. Only exists because
    it sometimes throws weird errors which require us to retry so that we don't skip any shows"""
    url = f'https://api.jikan.moe/v4/top/anime?page={page_num}'
    anime_batch = call_function_through_process(get_search_results, url)
    while anime_batch is None:
        # We keep retrying until we get the batch, the URL is correct so the main reason
        # for failure would be the API itself being down, in which case we just wait for it
        # to come back up.
        print("Anime batch was returned as None, sleeping and retrying")
        logger.debug("Anime batch was returned as None, sleeping and retrying")
        time.sleep(Sleep.MEDIUM_SLEEP)
        anime_batch = call_function_through_process(get_search_results, url)
    return anime_batch


def get_anime_batch_from_MAL(page_num, required_fields):

    url = f'https://api.myanimelist.net/v2/anime/ranking?ranking_type=all&limit=100&offset={page_num*100}&' \
          'fields='
    for field in required_fields:
        url = url + field + ","  # Concatenate all required fields with a comma inbetween
    url = url[:-1] # Remove the last comma
    anime_batch = get_search_results(url)
    # anime_batch = call_function_through_process(get_search_results, url)
    while anime_batch is None:
        # We keep retrying until we get the batch, the URL is correct so the main reason
        # for failure would be the API itself being down, in which case we just wait for it
        # to come back up.
        print("Anime batch was returned as None, sleeping and retrying")
        logger.debug("Anime batch was returned as None, sleeping and retrying")
        time.sleep(Sleep.MEDIUM_SLEEP)
        # anime_batch = call_function_through_process(get_search_results, url)
        anime_batch = get_search_results(url)
    return anime_batch


def count_scored_shows(user_list):
    count = 0
    for anime in user_list:
        count += 1
        if anime['list_status']['score'] == 0:
            return count
    return count


# @timeit
def get_user_MAL_list(user_name, full_list=True):
    """Helper function of fill_list_database. Gets the full list of one MAL user via their username.
    If full_list is false, it'll stop once it gets to the shows with no score"""
    url = f'https://api.myanimelist.net/v2/users/' \
          f'{user_name}/animelist?fields=list_status&limit=1000&sort=list_score&nsfw=True'

    # response = call_function_through_process(get_search_results, url)
    # response = get_search_results(url)
    response = get_data(url)
    anime_list = response["data"]
    # try:
    #     anime_list = response["data"]
    # except (TypeError, KeyError) as ex:
    #     anime_list = []  # list is empty/private
    #     print(response)
    #     print(response.status_code)
    #     print(ex)
    #     if response.status_code == 403:
    #         raise UserListPrivateError
    #     else:
    #         raise UserListFetchError

    # If the user has more than 1000 entries in their list, we will need separate API
    # calls for each thousand.

    if len(anime_list) == 1000:
        scored_shows = count_scored_shows(anime_list)
        thousands = 1
        while len(anime_list) == 1000 * thousands and (scored_shows == 1000 * thousands
                                                       or full_list):
            print(
                f'Length of {user_name}\'s anime list exceeds {1000 * thousands}, '  #
                f'proceeding to next 1000 shows')
            logger.debug(
                f'Length of {user_name}\'s anime list exceeds {1000 * thousands}, '
                f'proceeding to next 1000 shows')
            url = f'https://api.myanimelist.net/v2/users/' \
                  f'{user_name}/animelist?fields=list_status&limit=1000&sort' \
                  f'=list_score' \
                  f'&offset={1000 * thousands}&nsfw=True'  #
            response = get_data(url)
            next_part = response["data"]
            anime_list = anime_list + next_part
            thousands += 1
            scored_shows = count_scored_shows(anime_list)

    return anime_list




