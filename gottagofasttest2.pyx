
from __future__ import print_function
cimport numpy as np
from http import HTTPStatus
from multiprocessing import freeze_support
from datetime import datetime, date

# from Cython.Includes.libc.stdio import printf
import Cython
import cython
from dateutil.relativedelta import relativedelta
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
from enum import Enum
import GenerateAccessToken as AccessToken
import logging
import os
import csv
import sys
from requests.exceptions import SSLError, JSONDecodeError
import multiprocessing as mp
import shutil
from random import shuffle
import numpy.ma as ma

try:
    import thread
except ImportError:
    import _thread as thread
import urllib3
from hanging_threads import start_monitoring
from operator import itemgetter
from colorama import Fore
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from general_utils import *
from MAL_utils import *
import ctypes
from multiprocessing import Manager

logging.basicConfig(level=logging.WARNING, filename='Test.log', filemode='a',
                    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s") #
logger = logging.getLogger('MALRecommendations')
logger.setLevel(level=logging.DEBUG)

main_database_name = "ScoresDBFinal.csv"
temp_database_name = "TempScoresDBFinal.csv"


@timeit
def fill_main_database(amount, from_scratch=False, continuing=False, current_users=0,
                    MAL_user_list=None): #
    """ This function takes the anime DB created previously during the run of #
        create_main_database, takes the titles and IDs of shows and uses those
        title-ID pairs to create a url that links to the updates table of each show.
        From said updates table, it scrapes the usernames of the people who recently
        updated their list (and thus are known to be active) and adds their lists
        to the database.


        If from_scratch = False, it will try to get the usernames from an already existing
        file first before proceeding to scrape."""

    def add_user_list_to_db(user_index, user_name, user_list):
        """Takes a single user's list and creates a row for them in the main database."""

        user_scored_shows = count_scored_shows(user_list)
        if user_scored_shows >= 50:
            # User needs to have scored at least 50 shows to be part of the DB
            show_amount = 0
            score_sum = 0
            for anime in user_list:  # anime_list, scores_db
                title = anime['node']['title']
                score = anime['list_status']['score']
                # remove PTW for the total show count
                if score == 0:
                    print(f'Currently on show {show_amount} of user {user_name} ({user_index})')
                    print(f'Title is {title}')
                    break
                else:
                    show_amount += 1
                    if title in scores_db.columns:
                        scores_db.at[user_index, title] = score
                    score_sum += score
            scores_db.at[user_index, 'Username'] = user_name
            scores_db.at[user_index, 'Scored Shows'] = show_amount
            scores_db.at[user_index, 'Total Shows'] = len(user_list)
            scores_db.at[user_index, 'Mean Score'] = round(score_sum / show_amount, 4)
            return True
        print("User does not have enough scored shows, moving on to next user")
        return False

    scores_db = pd.read_csv(temp_database_name)
    # We use a temp database and only copy it to the real database in the end, so that if the
    # program unexpectedly crashes in the middle (several days' runtime), we don't lose our
    # previous database and can call continue_filling_database to continue from where we left off.

    if MAL_user_list is None:
        MAL_user_list = []

    # We'll want to save MAL usernames to avoid duplicates, so if the function didn't receive an
    # existing user list (such as from the continue_filling_database function), we'll initialize
    # it here.

    if os.path.exists("Blacklist.csv"):
        with open("Blacklist.csv", newline="", encoding='utf-8') as f:
            blacklist = next(csv.reader(f, delimiter=','))
    else:
        blacklist = []

    # The blacklist will contain anyone whose list was found to be too short/to have too
    # few scored shows. This will significantly save runtime, as we won't have to fetch their
    # lists if we run into their usernames again.

    if os.path.exists("NoSequelAnimeDB.csv"):  # It could be that only one of them exists
        anime_db_filename = "NoSequelAnimeDB.csv"
    else:
        anime_db_filename = "AnimeDB.csv"

    # We get the ids and titles of each show from the previously created shows database.

    with open(anime_db_filename, newline="", encoding='utf-8') as f:
        anime_db = csv.reader(f, delimiter=',')
        titles = (next(anime_db))[1:]  # Titles are the first row
        ids = (next(anime_db))[1:]  # IDs are the second row
        mean_scores = (next(anime_db))[1:]  # Mean scores are the third row
        scored_amount = (next(anime_db))[1:]  # Scored amount is the 4th row

    ids_titles_scored = list(zip(ids, titles, scored_amount))
    shuffle(ids_titles_scored)

    # Regarding shuffle - we don't want the iteration to always start with the first show on
    # the list, as it's by far one of the most popular shows. This means that a huge portion
    # of the people updating it will be new anime watchers with very few shows on their list.

    for id, title, scored_amount in ids_titles_scored:
        if current_users == amount:
            break
        # We reached the necessary amount of users in the database
        if int(scored_amount) < 10000:
            print(f"Scored amount of {title} is {scored_amount}, moving to next show")
            time.sleep(Sleep.SHORT_SLEEP)
            continue
        print(f"Scored amount of {title} is {scored_amount}, proceeding with current show")
        # If a show is too niche, we don't want to take usernames from its last updates
        # table. The main reason is that the accounts that update these extremely niche
        # shows will often be "collector" accounts - weird accounts who have literally every
        # single MAL entry on their list. Collecting their list data will result both in
        # inaccurate data (as the accounts are fake) and in HEAVY program slowdown, since their
        # lists contain 20000+ entries.
        title = replace_characters_for_url(title)
        base_url = f"https://myanimelist.net/anime/{id}/{title}/stats?"
        print(base_url)
        users_table = get_usernames_from_show(base_url)
        # This returns a table of 375 list updates

        for table_row in users_table:
            # The list update table includes usernames, timestamps and more.
            # We extract the usernames from there by their assigned CSS class.
            if current_users == amount:
                break
            user_link = table_row.findNext(
                "div", {"class": "di-tc va-m al pl4"}).findNext("a") #
            user_name = user_link.string
            if user_name not in MAL_user_list and user_name not in blacklist:
                # If user in list, we already have them in the database, no need to waste time.
                # If user in blacklist, their list is too short.
                user_anime_list = get_user_MAL_list(user_name, full_list=False)
                if len(user_anime_list) >= 100:
                    # To get the most out of our database, we'll only add people
                    # who have at least 100 entries. Furthermore, inside the add
                    # function, another filter checks if they have at least 50
                    # SCORED entries. If they do, "added" returns True.
                    added = add_user_list_to_db(current_users,
                                                user_name, user_anime_list) #
                    if added:
                        current_users += 1
                        MAL_user_list.append(user_name)
                    else:
                        blacklist.append(user_name)
            else:
                print(f"{user_name} is a duplicate user/list too short, moving on to next user")

        print(f"Saving database. Currently on {current_users} entries")
        scores_db.to_csv(temp_database_name)
        # with open("MALUsers.csv", "w", encoding='utf-8', newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(MAL_user_list)
        save_list_to_csv(MAL_user_list, "MALUsers.csv")
        save_list_to_csv(blacklist, "Blacklist.csv")
        logger.debug(f"User list length is : {len(MAL_user_list)}")
        logger.debug(f"Blacklist length is : {len(blacklist)}")

    # We save both the database and the user lists every once in a while instead of at the
    # end, so that we can continue more or less from where we finished if program stops.

    print(f'Final DB length : {current_users}')
    scores_db.to_csv(temp_database_name)
    print(scores_db)


def continue_filling_database(amount):
    temp_df = pd.read_csv(temp_database_name)
    current_users = len(temp_df.index)
    amount = amount - current_users
    print(amount)
    fill_main_database(amount, from_scratch=False, continuing=True, current_users=current_users,
                    MAL_user_list=temp_df['Username'].tolist()) #

@timeit
def create_main_database(amount_of_users, users_from_scratch=False, shows_from_scratch=False,
                        non_sequels_only=False): #
    """This is the main function. It creates a large CSV file of all the anime lists
    (with scores) of the people returned from get_MAL_usernames and get_RAL_usernames.

        Parameters :

            amount_of_users -
                The amount of users that we want in the database.

            users_from_scratch -
                True if we want to create a completely new database with a different
                set of MAL users. False if we want to use the existing MAL users file.

            shows_from_scratch -
                True if we want to renew the shows database (to include new seasonal
                shows), False if we want to use the existing shows database.

            non_sequels_only -
                True if we want only non-sequel shows to be in our shows database (will be
                used for recommendations), False if we want all rated entries on MAL.
    """

    def get_titles_from_animeDB(filename):  # Add this to helper functions module?
        if os.path.exists(filename):
            with open(filename, encoding='utf-8', newline="") as f1:
                titles = next(csv.reader(f1, delimiter=','))
                print(titles)
                return titles[1:]
        return None

    if shows_from_scratch:
        generate_anime_DB(non_sequels_only)

    if non_sequels_only:
        titles = get_titles_from_animeDB("NoSequelAnimeDB.csv")
        if not titles:
            # Database doesn't exist, shows_from_scratch was mistakenly
            # passed as False. Making it from scratch regardless.
            generate_anime_DB(non_sequels_only)
    else:
        titles = get_titles_from_animeDB("AnimeDB.csv")
        if not titles:
            generate_anime_DB(non_sequels_only)

    # Here we create the first row of the database, which includes
    # all the columns (all show names, all statistics, and the 'Usernames'
    # column.
    with open(temp_database_name, "w", encoding='utf-8', newline="") as f2:
        person_stats = ['Mean Score', 'Scored Shows', 'Total Shows']
        writer = csv.writer(f2)
        writer.writerow(['Username'] + person_stats + titles)
    # for username in MAL_user_list:
    #     writer.writerow([username])

    fill_main_database(amount_of_users, users_from_scratch)  # This actually fills temp database
    shutil.copy(temp_database_name, main_database_name)
# Once we finished filling our temp database, we can safely overwrite the main one.
# NOTE - terminating create_main_database and running it again WILL OVERWRITE
# THE TEMP DATABASE. To continue filling the database in the event of an unexpected
# program termination, use continue_filling_database(amount).


def generate_anime_DB(non_sequels_only=False):
    page_num = 1
    last_show_reached = False
    anime_data_list = [['', 'ID', 'Mean Score', 'Scores',
                        'Members', 'Favorites', 'Year', 'Season']] #
    # anime_data_list will be the list of all the DB entries (each of which is also
    # a list). It is initialized to have the column names as the first list (since those
    # will technically be the first "entry" in our database).
    last_show = 'Abunai Sisters: Koko & Mika'
    # Lowest rated show, it'll be the last one in the sorted list
    required_fields = ["title", "mal_id", "score",
                        "scored_by", "members", "favorites"] #
    # The fields we need from the JSON object containing information about a single anime.

    while not last_show_reached:
        # We loop over pages of anime info that we get from the Jikan API (the info is
        # sorted by score, from highest to lowest) until we reach the lowest rated show.
        # There should be 50 shows per page/batch.

        anime_data = []
        anime_batch = get_anime_batch_from_jikan(page_num)
        url = f'https://api.jikan.moe/v4/top/anime?page={page_num}'
        anime_batch = call_function_through_process(get_search_results, url)
        print(anime_batch)
        logging.info(anime_batch)
        while anime_batch is None:
            # If we failed to get the batch for some reason, we retry until it's a success.
            # Jikan API does not require authorization, so the only reason for a failure
            # could be an outage in the API itself, in which case we wouldn't want to
            # timeout/stop the function as an outage could technically last for a few hours,
            # or even days.
            print("Error - unable to get batch. Sleeping just to be safe, "
                "then trying again.") #
            logging.error("Error - unable to get batch. Sleeping just to be safe, "
                        "then trying again.") #
            time.sleep(Sleep.LONG_SLEEP)
            anime_batch = call_function_through_process(get_search_results, url)
            print(anime_batch)

        for anime in anime_batch["data"]:
            if not non_sequels_only or anime_is_valid(anime):
                anime_data = create_anime_DB_entry(anime, required_fields)
                anime_data_list.append(anime_data)
                logger.debug(anime_data)
            if anime["title"] == last_show:
                last_show_reached = True
                break
        # if break_flag == 1:  # lowest-rated TV show, once we've gotten it we're done
        #     # with our anime list - everything
        #     # after it will be either unrated shows or hentai.
        #     break
        page_num += 1
    anime_data_list = list(zip(*anime_data_list))
    logger.debug(anime_data_list)
    print(len(anime_data_list))
    df = pd.DataFrame(anime_data_list[1:], columns=anime_data_list[0])
    print(non_sequels_only)
    if not non_sequels_only:
        df_to_csv(df, "AnimeDB.csv")
    else:
        df_to_csv(df, "NoSequelAnimeDB.csv")


def user_list_to_df(user_list):
    # user_titles = []
    # user_scores = []
    dummy_db_filename = "AnimeDB.csv"
    dummy_db = pd.read_csv(dummy_db_filename)  #dummy DB containing just the show names
    col_names = dummy_db.columns
    new_row = pd.DataFrame(columns=col_names[1:])
    new_row.loc[0] = np.nan
    # print(f"Length of new row is {len(new_row)}")
    # print(f"Length of columns is {len(col_names)}")
    # print(f"First 10 col names : {col_names[0:10]}")
    # print(f"New row looks like this : {new_row}")
    df_to_csv(new_row, "TestNewRow.csv")

    i = 0
    score_sum = 0
    for anime in user_list:
        title = anime['node']['title']
        score = anime['list_status']['score']
        # remove PTW for the total show count
        if score == 0:
            break  # Lists are sorted by score, once we get to a 0 everything below
        # it is also 0
        else:
            i += 1
            # if i == 50:
            #     remove_from_db = False  # User needs to have scored at least 50 shows
            if title in new_row.columns:
                new_row[title] = score
            score_sum += score
    # else:
    # new_row['Mean Score'] = score_sum / i
    # new_row['Scored Shows'] = i
    # new_row['Total Shows'] = len(user_list)
    # print(new_row) #this will be added to main database in the final version
    return new_row


def compute_affinity(list1, list2):
    """Simply computes the Pearson's Correlation Coefficient between two vectors (arrays)."""
    return list1.corr(list2)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef double calculate_mean_without_zeros(np.ndarray[unsigned char] lst):
    cdef int list_len = len(lst)
    cdef double summ = 0
    cdef int count=0
    for i in range(list_len):
        if lst[i]!=0:
            summ += lst[i]
            count+=1
    return summ/count


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef double compute_affinity2(np.ndarray[unsigned char] list1, np.ndarray[unsigned char] list2, int list_len):
    """Simply computes the Pearson's Correlation Coefficient between two vectors (arrays)."""
    cdef double mean1 = calculate_mean_without_zeros(list1)
    cdef double mean2 = calculate_mean_without_zeros(list2)
    cdef double sum_x = 0
    cdef double sum_y = 0
    cdef double sum_x2 = 0
    cdef double sum_y2 = 0
    cdef double sum_xy = 0
    cdef int count=0
    for i in range(list_len):
        if list1[i]!=0 and list2[i]!=0:
            sum_xy+=list1[i]*list2[i]
            sum_y2+=list2[i]**2
            sum_x2+=list1[i]**2
            sum_x+=list1[i]
            sum_y+=list2[i]
            count+=1
    return (count*sum_xy - (sum_x*sum_y))/np.sqrt((count*sum_x2-sum_x**2)*(count*sum_y2-sum_y**2))




def count_common_shows(list1, list2):
    count = 0
    # print(f' List looks like the following : {list1}')
    # for show in list1:
    #     if not np.isnan(list1[show]) and not np.isnan(list2[show]):
    #         count+=1

    # for i in range(len(list1)):
    #     if not np.isnan(list1[i]) and not np.isnan(list2[i]):
    #         count += 1

    for i in range(len(list1)):
        if list1[i]!=0 and list2[i]!=0:
            count += 1

    # count = len([i for i in range(len(list1)) if list1[i] and list2[i]])
    return count

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef int count_common_shows2(np.ndarray[unsigned char] list1, np.ndarray[unsigned char] list2, int list_len):
    cdef int count = 0
    for i in range(list_len):
        if list1[i] and list2[i]:
            count += 1
    return count


@timeit
def find_max_affinity(username, amount=None):
    """This function iterates over rows in the database and calculates the affinity
    # between the user provided and each person (unless a limit is given) in the database.
    # The affinity is calculated via Pearson's Correlation Coefficient.

        Parameters :

            username - The MAL username of the user for whom the function tries to find
                    the user with the max affinity amongst the ones in the database

            amount - Maximum amount of people to check. If amount exceeds the actual
                    amount of users in the database, the entire database will be iterated
                    over.
    """

    # affinities_list.append((df.loc[user_index]['Username'], affinity, common_shows))

    min_common_shows = 20
    user_list = get_user_MAL_list(username, full_list=False)

    if user_list is None:
        print("Error - user does not exist, or site is currently down.")
    affinities_list = []
    user_list = user_list_to_df(user_list)
    # print(f"User list as returned from user_list_to_df : {user_list}")
    # df_to_csv(user_list, "TestList.csv")
    user_list = user_list.loc[0].astype('float64')
    print(user_list.array)
    # terminate_program()
    # print(f"User list after conversion to float64 : {user_list}")
    # print(f"Columns are : {user_list['Steins;Gate']}")
    # converting user scores to arrays for affinity calculation
    # terminate_program()
    main_df = pd.read_csv("TestScoresDBFinal.csv", index_col=[0])
    # print(f"The index is {df.index}")
    # print(len(df.index))

    for i in main_df.index:
        comparison_list = main_df.loc[i]['Kaguya-sama wa Kokurasetai: Ultra Romantic':] \
            .astype('float64') #
        # print(user_list)
        # print(comparison_list)

        # print(f"Common shows with user {main_df.loc[i]['Username']}")
        affinity = compute_affinity(user_list[1:1000], comparison_list[1:1000])
        if affinity >= 0.4:
            # common_shows = count_common_shows(user_list.array.fillna(0),
            #                                 comparison_list.array.fillna(0))
            common_shows = count_common_shows2(user_list.array.fillna(0),
                                            comparison_list.array.fillna(0),
                                            len(user_list))
            if common_shows > min_common_shows:
                affinity = compute_affinity(user_list, comparison_list)
                affinities_list.append((main_df.loc[i]['Username'], affinity, common_shows))
    # We create a list of tuples, with each tuple having a user from the DB
    # and his affinity % to the main user given to the function.
    print(affinities_list)
    for pair in affinities_list:
        if np.isnan(pair[1]):
            print(pair)
            affinities_list.remove(pair)

    for pair in sorted(affinities_list, reverse=True, key=lambda x: x[1]):
        print(pair)


if __name__ == '__main__':
    start_monitoring(seconds_frozen=Sleep.LONG_SLEEP + 10, test_interval=100)
    # This monitors threads that have been dormant for more than the longest sleep settings
    # allowed. Should not happen in practice since we're sending all our API calls through
    # separate process with a timeout of 15 seconds.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ser = Service("C:\\Program Files (x86)\\chromedriver.exe")
    op = webdriver.ChromeOptions()
    op.add_argument("--headless")
    op.add_argument("--lang=ja")
    op.add_argument('--blink-settings=imagesEnabled=false')
    driver = webdriver.Chrome(service=ser, options=op)

    BASE_PATH = "data"
    HTML_PATH = BASE_PATH + "/html"
    USER_PATH = BASE_PATH + "/users"
    freeze_support()
    find_max_affinity("BaronBrixius")
