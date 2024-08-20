from __future__ import print_function

# from main.modules.AnimeList import AnimeList
from main.modules.general_utils import get_data, Sleep, convert_to_timestamp
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

    status_convert_dict = {'COMPLETED': 'completed', 'DROPPED': 'dropped',
                   'CURRENT': 'watching', 'PAUSED': 'on_hold',
                   'PLANNING': 'plan_to_watch', 'REPEATING': 'completed'}  # No MAL equivalent for repeating

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
        for anime_data in user_list.list_obj:
            count += 1
            if anime_data['list_status']['score'] == 0:
                return count
        return count

    @staticmethod
    def get_anime_batch_from_MAL(page_num, required_fields):

        url = f'https://api.myanimelist.net/v2/anime/ranking?ranking_type=all&limit=100&offset={page_num * 100}&' \
              'fields='
        for field in required_fields:
            url = url + field + ","
        url = url[:-1]
        anime_batch = get_data(url)
        while anime_batch is None:
            time.sleep(5)
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
                # Something wrong with the page html, very rare, easier to just skip the user than handle it

        soup = BeautifulSoup(page_html.text, "html.parser")
        date_str = soup.findAll("span", {"class": "user-status-data di-ib fl-r"})[
            -1].text  # We'll always need the last one
        date_obj = datetime.datetime.strptime(date_str, "%b %d, %Y")
        account_creation_time = date_obj.replace(tzinfo=datetime.timezone.utc)
        current_time = datetime.datetime.now(datetime.timezone.utc)
        time_since_account_creation = current_time - account_creation_time
        return time_since_account_creation

    @staticmethod
    def get_usernames_from_show(base_url, pages_to_get=10):
        """This function takes an anime title, accesses the updates table on the Stats
        tab on the MAL page of that title, and returns pages_to_get*75 usernames from it.
        This is the easiest way to gather recently active usernames."""
        users_table = []
        too_many_requests_flag = 0
        page_num = 0
        while page_num < pages_to_get:
            page_users_table = []
            url = f"{base_url}show={str(75 * page_num)}"
            print(f"Getting usernames from {url}")
            # logger.debug(f"Getting usernames from {url}")
            page_html = get_data(url)
            time.sleep(1)
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
    # move to AnilistUtils?
    def convert_anilist_status_to_MAL(cls, anilist_status):
        return cls.status_convert_dict[anilist_status]

    @staticmethod
    def MAL_user_time_string_to_timestamp(str):
        # seconds_to_substract = None
        if "ago" in str:
            seconds_to_substract, units, _ = str.split(" ")
            seconds_to_substract = int(seconds_to_substract)
            if "minute" in units:
                seconds_to_substract *= 60
            elif "hour" in units:
                seconds_to_substract *= 3600
            date_timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=seconds_to_substract)).timestamp()

        else:
            user_date, user_time = str.split(",")
            user_time = user_time.strip()
            if "day" in user_date:
                date_timestamp = datetime.datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(hours=10)
                if user_date == "Yesterday":
                    date_timestamp = date_timestamp - datetime.timedelta(days=1)
                # date_obj = datetime.datetime.fromtimestamp(date_timestamp.timestamp())

                # Format the datetime object to a string in the desired format
                user_date = date_timestamp.strftime("%b %d")

            date_timestamp = convert_to_timestamp(user_date, user_time.strip())

        return date_timestamp

    @staticmethod
    def get_recent_ptw_data(start_date, end_date):
        start_date, start_time = start_date.split(",")
        end_date, end_time = end_date.split(",")

        start_timestamp = convert_to_timestamp(start_date, start_time.strip())
        end_timestamp = convert_to_timestamp(end_date, end_time.strip())
        ptw_count = 0
        url = "https://myanimelist.net/anime/55102/Girls_Band_Cry/stats?m=all&"
        users_table = MALUtils.get_usernames_from_show(url, pages_to_get=100)
        for (i, table_row) in enumerate(users_table):
            # user_link = table_row.findNext(
            #     "div", {"class": "di-tc va-m al pl4"}).findNext("a")
            # user_name = str(user_link.string)

            td_elements = table_row.find_all("td", {"class": "borderClass ac"})
            if len(td_elements) >= 3:
                user_list_status = str(td_elements[1].string)
                user_time_string = str(td_elements[3].string)
                user_timestamp = MALUtils.MAL_user_time_string_to_timestamp(user_time_string)
                if user_list_status == "Plan to Watch" and (start_timestamp <= user_timestamp
                                                            ) and user_timestamp <= end_timestamp:
                    ptw_count += 1



        return ptw_count




