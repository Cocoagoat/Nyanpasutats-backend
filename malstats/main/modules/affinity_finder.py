from __future__ import print_function
from time import perf_counter
try:
    import thread
except ImportError:
    import _thread as thread
from MAL_utils import Data
from general_utils import time_at_current_point, timeit
import numpy as np
import pyximport # Need this to import .pyx files
pyximport.install(setup_args = {"include_dirs":np.get_include()}) # Need to manually include numpy or it won't work
import gottagofasttest2


@timeit
def find_max_affinity(username, min_common_shows=20):
    """This function iterates over rows in the database and calculates the affinity
    # between the user provided and each person (unless a limit is given) in the database.
    # The affinity is calculated via Pearson's Correlation Coefficient.

        Parameters :

            username - The MAL username of the user for whom the function tries to find
                       the user with the max affinity amongst the ones in the database

            amount - Maximum amount of people to check. If amount exceeds the actual
                     amount of users in the database, the entire database will be iterated
                     over."""

    def initial_user_prompt():
        while(True):
            x = input(f"Current database size is {data.main_df.shape[0]}. Continue? Y/N")
            if x=='Y':
                return
            if x=='N':
                amount = int(input("Insert new database size"))
                data.fill_main_database(amount)

    data = Data()
    # initial_user_prompt()

    t1 = perf_counter()
    username_list = list(data.scores_dict.keys())
    try:
        user_list = data.scores_dict[username]
    except KeyError:
        user_list = data.create_user_entry(username)
    affinities_list = []

    time_at_current_point(t1, "Start")
    print("Starting calculations")

    for i in range(len(data.scores_dict)):
        # This loop calculates the affinity between the main user and every score
        # list in the dictionary. This is done through Cythonized functions which
        # dramatically speed up the process (a must since we have 100k+ users to compare with)

        comparison_list = data.scores_dict[username_list[i]]
        common_shows = gottagofasttest2.count_common_shows2(user_list, comparison_list, len(user_list))

        if common_shows > min_common_shows:
            affinity = gottagofasttest2.compute_affinity2(user_list, comparison_list, len(user_list))
            affinities_list.append((username_list[i], affinity, common_shows))

    for pair in affinities_list:
        if np.isnan(pair[1]) or pair[1]==1:
            affinities_list.remove(pair)
            # Affinity of 1 = the main user is being compared to themselves, it's easier to remove
            # the duplicate here than check for user name every time when calculating. nan means
            # that it's mathematically impossible to calculate the affinity (denominator is 0).

    for pair in sorted(affinities_list, reverse=True, key=lambda x: x[1]):
        if abs(pair[1])>0.6 or abs(pair[1])==0:
            print(pair)

    affinity_percentages = [100*x[1] for x in affinities_list]
    amount_of_people = len(affinities_list)
    positive_affinity = len([x for x in affinity_percentages if x > 0])
    negative_affinity = len([x for x in affinity_percentages if x < 0])
    zero_affinity = amount_of_people - positive_affinity - negative_affinity

    print(f"Your average affinity is : {round(np.mean(affinity_percentages),2)}%")
    print(f"Positive affinity : {positive_affinity},{round(100 * positive_affinity / amount_of_people, 2)}%")
    print(f"Negative affinity : {negative_affinity},{round(100 * negative_affinity / amount_of_people, 2)}%")
    print(f"Zero affinity count : {zero_affinity},{round(100 * zero_affinity/ amount_of_people, 2)}%")
