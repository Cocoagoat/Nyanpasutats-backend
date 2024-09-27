from __future__ import print_function
from time import perf_counter
from main.modules.AnimeListHandler import AnimeListHandler
from main.modules.UserDB import UserDB
try:
    import thread
except ImportError:
    import _thread as thread
from main.modules.general_utils import time_at_current_point, timeit, list_to_uint8_array, load_pickled_file
import numpy as np
from main.modules.filenames import scores_dict_filename, aff_db_path

import pyximport  # Need this to import .pyx files
pyximport.install(setup_args={"include_dirs": np.get_include()})  # Need to manually include numpy or it won't work
from main.modules.gottagofasttest2 import count_common_shows, compute_affinity


def find_max_affinity(username, site="MAL", min_common_shows=20):
    """This function iterates over rows in the database and calculates the affinity
    between the user provided and each person (unless a limit is given) in the database.
    The affinity is calculated via Pearson's Correlation Coefficient.

    :param username - The MAL username of the user for whom the function tries to find
                       the user with the max affinity amongst the ones in the database.

    :param site - The scoring site used (MAL/Anilist/etc)

    :param min_common_shows - The minimum common shows needed for the person to be included
    in the final statistics. 20 is the default used by MAL.
    """

    ListHandler = AnimeListHandler.get_concrete_handler(site)
    user_list = ListHandler(username).get_user_scores_list()
    user_list = list_to_uint8_array(user_list)
    user_list_len = len(user_list)

    affinities_list = []
    pos_aff_count = 0
    neg_aff_count = 0
    avg_aff = 0

    filename = scores_dict_filename.split(".")[0]
    if str(aff_db_path).startswith("/mnt") or 'home' in str(aff_db_path):  # Linux support
        symbol = "/"
    else:
        symbol = "\\"

    for i in range(99):
        scores_dict = {}
        for _ in range(300):
            # Retry until there's no conflict between processes
            try:
                scores_dict = load_pickled_file(f"{aff_db_path}{symbol}{filename}-P{i+1}.pickle")
                break
            except FileNotFoundError:
                user_db = UserDB()
                user_db.fill_main_database(150000)
            except OSError:
                continue

        if not scores_dict:
            raise OSError("Unable to access scores database. Please try again later.")

        username_list = list(scores_dict.keys())
        for j in range(len(scores_dict)):
            # This loop calculates the affinity between the main user and every score
            # list in the dictionary. This is done through Cythonized functions which
            # dramatically speed up the process (a must since we have 100k+ users to compare with)
            if username_list[j] == username:
                continue
            comparison_list = scores_dict[username_list[j]]
            common_shows = count_common_shows(user_list, comparison_list, user_list_len)

            if common_shows > min_common_shows:
                affinity = compute_affinity(user_list, comparison_list, user_list_len)
                if np.isnan(affinity):
                    continue
                affinities_list.append({'Username': username_list[j],
                                        'Affinity': round(affinity*100, 2),
                                        'CommonShows': common_shows})

                if affinity > 0:
                    pos_aff_count += 1
                elif affinity < 0:
                    neg_aff_count += 1
                avg_aff += affinity*100

    user_count = len(affinities_list)
    try:
        avg_aff /= user_count
    except ZeroDivisionError:
        avg_aff = 0
    sorted_affinities_list = sorted(affinities_list, reverse=True, key=lambda x: x['Affinity'])
    # sorted_aff_dict_items = sorted(affinities_list.items(), reverse=True, key=lambda x: x[1]['Affinity'])
    # pos_affinities = {str(user): user_dict for user, user_dict in sorted_aff_dict_items[0:500]}
    # neg_affinities = {str(user): user_dict for user, user_dict in sorted_aff_dict_items[-1:-501:-1]}
    pos_affinities = sorted_affinities_list[0:500]
    neg_affinities = sorted_affinities_list[-1:-501:-1]
    return pos_affinities, neg_affinities
