from __future__ import print_function
from time import perf_counter

from main.modules.AnimeListHandler import MALListHandler, AnilistHandler

try:
    import thread
except ImportError:
    import _thread as thread
from main.modules.general_utils import time_at_current_point, timeit, list_to_uint8_array
import numpy as np
from collections import defaultdict
from main.modules.UserDB import UserDB

import pyximport # Need this to import .pyx files
pyximport.install(setup_args = {"include_dirs":np.get_include()}) # Need to manually include numpy or it won't work
from main.modules.gottagofasttest2 import count_common_shows, compute_affinity


@timeit
def find_max_affinity(username, site="MAL", min_common_shows=20):
    """This function iterates over rows in the database and calculates the affinity
    between the user provided and each person (unless a limit is given) in the database.
    The affinity is calculated via Pearson's Correlation Coefficient.

        Parameters :

            username - The MAL username of the user for whom the function tries to find
                       the user with the max affinity amongst the ones in the database

            amount - Maximum amount of people to check. If amount exceeds the actual
                     amount of users in the database, the entire database will be iterated
                     over."""

    # def initial_user_prompt():
    #     while(True):
    #         x = input(f"Current database size is {user_db.main_df.shape[0]}. Continue? Y/N")
    #         if x == 'Y':
    #             return
    #         if x == 'N':
    #             amount = int(input("Insert new database size"))
    #             user_db.fill_main_database(amount)

    user_db = UserDB()

    t1 = perf_counter()
    username_list = list(user_db.scores_dict.keys())

    # user_list = None
    if site == "MAL":
        try:
            user_list = user_db.scores_dict[username]
            # Maybe user's already in database and we can save time
            # Remove this later cause outdated
        except KeyError:
            user_list = MALListHandler(username).get_user_scores_list()
    else:
        user_list = AnilistHandler(username).get_user_scores_list()

    user_list = list_to_uint8_array(user_list)
    # if user_list is None:
    #     user_list =
    #     #change this to new style
    affinities_list = defaultdict()

    time_at_current_point(t1, "Start")
    print("Starting calculations")
    pos_aff_count = 0
    neg_aff_count = 0
    avg_aff = 0

    for i in range(len(user_db.scores_dict)):
        # This loop calculates the affinity between the main user and every score
        # list in the dictionary. This is done through Cythonized functions which
        # dramatically speed up the process (a must since we have 100k+ users to compare with)
        if username_list[i] == username:
            continue
        comparison_list = user_db.scores_dict[username_list[i]]
        common_shows = count_common_shows(user_list, comparison_list, len(user_list))

        if common_shows > min_common_shows:
            affinity = compute_affinity(user_list, comparison_list, len(user_list))
            if np.isnan(affinity):
                continue
            affinities_list[username_list[i]] = {'Affinity': round(affinity*100,2), 'CommonShows': common_shows}

            if affinity > 0:
                pos_aff_count += 1
            elif affinity < 0:
                neg_aff_count += 1
            avg_aff += affinity*100

    user_count = len(affinities_list)
    avg_aff /= user_count

    sorted_aff_dict_items = sorted(affinities_list.items(), reverse=True, key=lambda x: x[1]['Affinity'])
    pos_affinities = {user: user_dict for user, user_dict in sorted_aff_dict_items[0:50]}
    neg_affinities = {user: user_dict for user, user_dict in sorted_aff_dict_items[-1:-51:-1]}
    zero_affinity = user_count - pos_aff_count - neg_aff_count

    print(f"Your average affinity is : {avg_aff}%")
    print(f"Positive affinity : {pos_aff_count},{round(100 * pos_aff_count / user_count, 2)}%")
    print(f"Negative affinity : {neg_aff_count},{round(100 * neg_aff_count / user_count, 2)}%")
    print(f"Zero affinity count : {zero_affinity},{round(100 * zero_affinity/ user_count, 2)}%")

    return pos_affinities, neg_affinities
