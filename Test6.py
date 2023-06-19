import os

from general_utils import load_pickled_file
from Test2 import *
from MAL_utils import get_anime_batch_from_jikan
from MAL_utils import MediaTypes
from MAL_utils import Data
from Test7 import Tags
import requests
import json
import random
from affinity_finder import find_max_affinity
from general_utils import time_at_current_point


tags_db_filename = "UserTagDBTest"
# def calculate_affinities1(user_row):
#     for show, score in user_row.items():
#         if user_row[show]:
#             user_score = user_row[show]
#             # user_stds = (user_score - user_mean_score)/user_std_of_watched
#
#             show_score = mean_score_row[show].item()
#             show_stds = (show_score - MAL_mean_of_watched)/MAL_std_of_watched
#             # diff_user_mean = round(user_score - user_mean_score,3)
#             # diff_show_mean = round(user_score - show_score,3)
#             show_tags_list = shows_tags_dict[show]['Tags']
#             show_genres_list = shows_tags_dict[show]['Genres']
#             show_studio = shows_tags_dict[show]['Studio']
#
#             user_diff = (user_score - user_mean_score)  #*abs((5.5 - user_score)/4.5) #fix to min(0.5,?)
#             MAL_diff = (MAL_mean_of_watched - show_score)  #*abs((5.5 - show_score)/4.5)
#             MAL_diff_adjusted = min(MAL_diff,max(user_diff+1, -user_diff+1))
#             if show_score > MAL_mean_of_watched:
#                 MAL_diff_adjusted = MAL_diff_adjusted/2
#             else:
#                 MAL_diff_adjusted = MAL_diff_adjusted*2
#
#             affinity = (user_diff + MAL_diff_adjusted)*abs(user_diff)
#
#             if affinity < 0:
#                 if user_score==10:
#                     affinity = np.sqrt(affinity+1)
#                 if user_score==9 and user_mean_score < 9:
#                     continue
#             else: #If affinity is positive, but the score is extremely low and the user's mean score is higher,
#                 # disregard the calculation. This will only happen very rarely, mostly with shows that have a very
#                 # low score on MAL.
#                 match user_score:
#                     case 1 | 2 | 3 | 4:
#                         if user_mean_score > user_score:
#                             continue
#                     case 10:
#                         affinity = (affinity + 1) ** 3
#             if show_score<5 or user_score<5:
#                 continue # Troll score
#
#
#             sign = np.sign(affinity)
#             if abs(affinity)>1:
#                 # if affinity < 0:
#                 #     affinity = -np.sqrt(-1 * affinity)
#                 # else:
#                 #     affinity = np.sqrt(affinity)
#
#                 affinity = sign * np.sqrt(sign*affinity)
#             else:
#                 affinity = sign * (affinity**2)
#
#             affinity = min(10, affinity)
#
#             k=10
#
#             for tag in show_tags_list:
#                 if tag['percentage'] >= 70:
#                     coeff = round((tag['percentage']/100)**3,3)
#                     user_tag_affinity_dict[user_name][tag['name']] += \
#                         affinity * coeff
#                     user_tag_counts[tag['name']] += coeff
#                     user_tag_shows[tag['name']].append(show)
#                     user_tag_shows[tag['name']].append(affinity)
#                     user_tag_shows[tag['name']].append(user_score)
#                     user_tag_shows[tag['name']].append(show_score)
#                     user_tag_shows[tag['name']].append(user_diff)
#                     user_tag_shows[tag['name']].append(MAL_diff_adjusted)
#
#             for genre in show_genres_list:
#                 user_tag_affinity_dict[user_name][genre] += \
#                     affinity
#                 user_tag_counts[genre] += 1
#                 user_tag_shows[genre].append(show)
#                 user_tag_shows[genre].append(affinity)
#
#
#             # print(5)
#             if show_studio in all_tags_list:
#                 user_tag_affinity_dict[user_name][show_studio] += affinity
#                 user_tag_counts[show_studio] += 1
#                 user_tag_shows[show_studio].append(show)
#                 user_tag_shows[show_studio].append(affinity)
#
#
# def calculate_affinities2(user_row, user_mean_score):
#
#     threshold_score = min(round(user_mean_score + 1.5), 10)
#     # exp_coeffs = [2 ** (score - threshold_score) for score in range(10, threshold_score-1, -1)]
#     exp_coeffs = [2**(score-threshold_score) for score in range(threshold_score,11)]
#     show_count = 0
#
#     for show, score in user_row.items():
#         if user_row[show]:
#
#             user_score = user_row[show]
#             show_score = mean_score_row[show].item()
#
#             if user_score < threshold_score:
#                 continue
#
#             show_tags_list = shows_tags_dict[show]['Tags']
#             show_genres_list = shows_tags_dict[show]['Genres']
#             show_studio = shows_tags_dict[show]['Studio']
#
#             exp_coeff =  1/exp_coeffs[10-user_score]
#
#             for tag in show_tags_list:
#                 if tag['percentage'] >= 85:
#                     tag_coeff = round((tag['percentage']/100),3)
#                 elif tag['percentage']>=70:
#                     second_coeff = 23/6 - tag['percentage']/30 # Can be a number from 1 to 1.5
#                     tag_coeff = round((tag['percentage']/100)**second_coeff,3)
#                 elif tag['percentage']>=60:
#                     second_coeff = 19 - tag['percentage']/4
#                     tag_coeff = round((tag['percentage']/100)**second_coeff,3)
#                     # 70 ---> 1.5, 60 ---> 4, 10 forward means -2.5 so m = -1/4 y = -(1/4)x+19
#                 else:
#                     break
#                 user_tag_affinity_dict[user_name][tag['name']] += exp_coeff * tag_coeff
#
#
#             for genre in show_genres_list:
#                 user_tag_affinity_dict[user_name][genre] += exp_coeff
#                 # user_tag_counts[genre] += 1
#                 # user_tag_shows[genre].append(show)
#                 # user_tag_shows[genre].append(affinity)
#
#
#             # print(5)
#             if show_studio in all_tags_list:
#                 user_tag_affinity_dict[user_name][show_studio] += exp_coeff
#                 # user_tag_counts[show_studio] += 1
#                 # user_tag_shows[show_studio].append(show)
#                 # user_tag_shows[show_studio].append(affinity)
#
#             show_count += exp_coeff
#
#     for tag, affinity_value in user_tag_affinity_dict[user_name].items():
#         user_tag_affinity_dict[user_name][tag] = user_tag_affinity_dict[user_name][tag]/show_count
#
#
# def calculate_affinities3(user_row, user_mean_score):
#     """ How likely is this show to be among the user's favorites?"""
#
#     threshold_score = min(round(user_mean_score + 1.5), 10)
#     # exp_coeffs = [2 ** (score - threshold_score) for score in range(10, threshold_score-1, -1)]
#     exp_coeffs = [2 ** (score - threshold_score) for score in range(threshold_score, 11)]
#     show_count = 0
#     fav_show_count = 0
#     user_fav_tag_counts = {tag_name: 0 for tag_name in all_tags_list}
#     tag_fav_show_list = {tag_name : [] for tag_name in all_tags_list}
#     tag_show_list = {tag_name : [] for tag_name in all_tags_list}
#
#     for show, score in user_row.items():
#         if user_row[show]:
#
#             user_score = user_row[show]
#             show_score = mean_score_row[show].item()
#
#             if show_score <5:
#                 continue # Meme shows, we don't want to base decisions on those
#
#             show_count += 1
#             show_score_coeff = (-3/8)*show_score + 31/8
#
#
#             show_tags_list = shows_tags_dict[show]['Tags']
#             show_genres_list = shows_tags_dict[show]['Genres']
#             show_studio = shows_tags_dict[show]['Studio']
#
#
#             if user_score >= threshold_score:
#                 exp_coeff = 1 / exp_coeffs[10 - user_score]
#             else:
#                 exp_coeff = 0
#
#
#
#             for tag in show_tags_list:
#                 if tag['percentage'] >= 85:
#                     tag_coeff = round((tag['percentage'] / 100), 3)
#                 elif tag['percentage'] >= 70:
#                     second_coeff = 23 / 6 - tag['percentage'] / 30  # Can be a number from 1 to 1.5
#                     tag_coeff = round((tag['percentage'] / 100) ** second_coeff, 3)
#                 elif tag['percentage'] >= 60:
#                     second_coeff = 19 - tag['percentage'] / 4
#                     tag_coeff = round((tag['percentage'] / 100) ** second_coeff, 3)
#                     # 70 ---> 1.5, 60 ---> 4, 10 forward means -2.5 so m = -1/4 y = -(1/4)x+19
#                 else:
#                     break
#
#                 if user_score >= threshold_score:
#                     user_fav_tag_counts[tag['name']] += round(exp_coeff * tag_coeff * show_score_coeff,3)
#                     tag_fav_show_list[tag['name']].append(show)
#                 user_tag_counts[tag['name']] += round(tag_coeff,3)
#                 tag_show_list[tag['name']].append(show)
#
#             for genre in show_genres_list:
#                 if user_score >= threshold_score:
#                     user_fav_tag_counts[genre] += round(exp_coeff * show_score_coeff,3)
#                     tag_fav_show_list[genre].append(show)
#                 user_tag_counts[genre] += 1
#                 tag_show_list[genre].append(show)
#
#             if show_studio in all_tags_list:
#                 if user_score >= threshold_score:
#                     user_fav_tag_counts[show_studio] += round(exp_coeff * show_score_coeff,3)
#                     tag_fav_show_list[show_studio].append(show)
#                 user_tag_counts[show_studio] += 1
#                 tag_show_list[show_studio].append(show)
#
#             fav_show_count+=exp_coeff*show_score_coeff
#
#     for tag, count in user_fav_tag_counts.items():
#         tag_fav_ratio= count/fav_show_count
#         tag_overall_ratio = user_tag_counts[tag]/show_count
#         try:
#             fav_coeff = tag_fav_ratio/tag_overall_ratio
#             if fav_coeff>1:
#                 affinity = min(1 - 1/fav_coeff, tag_overall_ratio*20) #change 10 to a better % later
#             else:
#                 affinity = max(-1 + fav_coeff, -1*tag_overall_ratio*20)
#         except ZeroDivisionError:
#             affinity = 0 # Change this to normalized affinity lately
#
#         user_tag_affinity_dict[user_name][tag] = [affinity, tag_fav_show_list[tag],tag_show_list[tag]]


# def calculate_affinities4(user_scores,user_mean_score):
#     # data = Data()
#     # tags = Tags()
#     show_count=0
#
#     user_mean_score = partial_main_df.row(user_index, named=True)['Mean Score']
#     user_name = partial_main_df['Username'][user_index]
#
#     user_scores = partial_main_df.select(relevant_shows).row(user_index, named=True)
#     watched_user_score_list = [score for title, score in user_scores.items() if score != None]
#
#     watched_titles = [title for title, score in user_scores.items() if score]
#     watched_MAL_score_dict = mean_score_row.select(watched_titles).to_dict(as_series=False)
#     watched_MAL_score_list = [score[0] for title, score in
#                               watched_MAL_score_dict.items() if title != "Rows"]
#
#     MAL_mean_of_watched = np.mean(watched_MAL_score_list)
#     user_mean_of_watched = np.mean(watched_user_score_list)
#     user_tag_affinity_dict[user_name] = {tag_name: 0 for tag_name in tags.all_tags_list}
#
#
#     user_tag_counts = {tag_name: 0 for tag_name in tags.all_tags_list}
#
#     MAL_score_per_tag = {tag_name: 0 for tag_name in tags.all_tags_list}
#     user_score_per_tag = {tag_name: 0 for tag_name in tags.all_tags_list}
#
#
#     tag_show_list = {tag_name: [] for tag_name in tags.all_tags_list}
#     user_show_list=[]
#
#     # user_tag_shows = {tag_name: [] for tag_name in tags.all_tags_list}
#     # shows_per_tag = {tag_name: 0 for tag_name in all_tags_list}
#
#     for show, score in user_scores.items():
#         if user_scores[show]:
#
#             user_show_list.append(show)
#
#             user_score = user_scores[show]
#             show_score = mean_score_row[show].item()
#
#             if show_score <5:
#                 continue # Meme shows, we don't want to base decisions on those
#
#             show_count += 1
#             show_score_coeff = (-3/8)*show_score + 31/8
#
#             show_tags_list = shows_tags_dict[show]['Tags']
#             show_genres_list = shows_tags_dict[show]['Genres']
#             show_studio = shows_tags_dict[show]['Studio']
#
#             for tag in show_tags_list:
#                 if tag['percentage'] >= 85:
#                     tag_coeff = round((tag['percentage'] / 100), 3)
#                 elif tag['percentage'] >= 70:
#                     second_coeff = 23 / 6 - tag['percentage'] / 30  # Can be a number from 1 to 1.5
#                     tag_coeff = round((tag['percentage'] / 100)**second_coeff, 3)
#                 elif tag['percentage'] >= 60:
#                     second_coeff = 19 - tag['percentage'] / 4
#                     tag_coeff = round((tag['percentage'] / 100)**second_coeff, 3)
#                     # 70 ---> 1.5, 60 ---> 4, 10 forward means -2.5 so m = -1/4 y = -(1/4)x+19
#                 else:
#                     tag_coeff = 0
#                     break
#
#                 # print(tag)
#                 user_score_per_tag[tag['name']] += user_score*tag_coeff
#                 MAL_score_per_tag[tag['name']] += show_score*tag_coeff
#                 user_tag_counts[tag['name']] += tag_coeff
#
#                 tag_show_list[tag['name']].append(show)
#
#             for genre in show_genres_list:
#                 user_score_per_tag[genre] += user_score
#                 MAL_score_per_tag[genre] += show_score
#                 user_tag_counts[genre] += 1
#                 tag_show_list[genre].append(show)
#
#             if show_studio in all_tags_list:
#                 user_score_per_tag[show_studio] += user_score
#                 MAL_score_per_tag[show_studio] += show_score
#                 user_tag_counts[show_studio] += 1
#                 tag_show_list[show_studio].append(show)
#
#     for tag in all_tags_list:
#         try:
#             tag_overall_ratio = user_tag_counts[tag] / show_count
#             freq_coeff = min(1,max(user_tag_counts[tag]/10,tag_overall_ratio*20))
#             # User has to watch either at least 10 shows with the tag or have the tag in at least 5% of their watched
#             # shows for it to count fully.
#             user_tag_diff = user_score_per_tag[tag]/user_tag_counts[tag] - user_mean_of_watched
#             MAL_tag_diff = MAL_score_per_tag[tag]/user_tag_counts[tag] - MAL_mean_of_watched
#             user_tag_affinity_dict[user_name][tag] = (2*user_tag_diff - MAL_tag_diff)*freq_coeff
#         except ZeroDivisionError:
#             user_tag_affinity_dict[user_name][tag] = 0
#
#     user_tag_affinity_dict[user_name]['Shows'] = user_show_list


# def calculate_affinities5(user_row, mean_score):
#     show_count = 0
#
#     shows_per_tag = {tag_name: 0 for tag_name in all_tags_list}
#     MAL_score_per_tag = {tag_name: 0 for tag_name in all_tags_list}
#     user_score_per_tag = {tag_name: 0 for tag_name in all_tags_list}
#
#     tag_show_list = {tag_name: [] for tag_name in all_tags_list}
#
#     for show, score in user_row.items():
#         if user_row[show]:
#
#             user_score = user_row[show]
#             show_score = mean_score_row[show].item()
#
#             if show_score < 5:
#                 continue  # Meme shows, we don't want to base decisions on those
#
#             show_count += 1
#             show_score_coeff = (-3 / 8) * show_score + 31 / 8
#
#             show_tags_list = shows_tags_dict[show]['Tags']
#             show_genres_list = shows_tags_dict[show]['Genres']
#             show_studio = shows_tags_dict[show]['Studio']
#
#             for tag in show_tags_list:
#                 if tag['percentage'] >= 85:
#                     tag_coeff = round((tag['percentage'] / 100), 3)
#                 elif tag['percentage'] >= 70:
#                     second_coeff = 23 / 6 - tag['percentage'] / 30  # Can be a number from 1 to 1.5
#                     tag_coeff = round((tag['percentage'] / 100), 3)
#                 elif tag['percentage'] >= 60:
#                     second_coeff = 19 - tag['percentage'] / 4
#                     tag_coeff = round((tag['percentage'] / 100), 3)
#                     # 70 ---> 1.5, 60 ---> 4, 10 forward means -2.5 so m = -1/4 y = -(1/4)x+19
#                 else:
#                     tag_coeff = 0
#                     break
#
#                 print(tag)
#                 user_score_per_tag[tag['name']] += user_score * tag_coeff
#                 MAL_score_per_tag[tag['name']] += show_score * tag_coeff
#                 user_tag_counts[tag['name']] += tag_coeff
#
#                 tag_show_list[tag['name']].append(show)
#
#             for genre in show_genres_list:
#                 user_score_per_tag[genre] += user_score
#                 MAL_score_per_tag[genre] += show_score
#                 user_tag_counts[genre] += 1
#                 tag_show_list[genre].append(show)
#
#             if show_studio in all_tags_list:
#                 user_score_per_tag[show_studio] += user_score
#                 MAL_score_per_tag[show_studio] += show_score
#                 user_tag_counts[show_studio] += 1
#                 tag_show_list[show_studio].append(show)
#
#     for tag in all_tags_list:
#         try:
#             tag_overall_ratio = user_tag_counts[tag] / show_count
#             freq_coeff = min(1, max(user_tag_counts[tag] / 10, tag_overall_ratio * 20))
#             # User has to watch either at least 10 shows with the tag or have the tag in at least 5% of their watched
#             # shows for it to count fully. If user watched less than that, the average score for the tag will be taken
#             # as a weighted average between their mean score for the tag and their overall mean score.
#             # This will prevent stuff like someone only watching a single CGDCT show, giving it a 1/10 -
#             # and suddenly the model thinks he will hate every CGDCT show even if the rest of the tags fit their tastes.
#
#             user_score_per_tag[tag] = (1-freq_coeff)*(user_mean_of_watched)\
#                                       + freq_coeff*(user_score_per_tag[tag]/user_tag_counts[tag])
#             MAL_score_per_tag[tag] = (1-freq_coeff)*(MAL_mean_of_watched)\
#                                       + freq_coeff*(MAL_score_per_tag[tag]/user_tag_counts[tag])
#             user_tag_affinity_dict[user_name][tag] = [user_score_per_tag[tag], MAL_score_per_tag[tag]]
#             # user_tag_diff = user_score_per_tag[tag] / user_tag_counts[tag] - user_mean_of_watched
#             # MAL_tag_diff = MAL_score_per_tag[tag] / user_tag_counts[tag] - MAL_mean_of_watched
#             # user_tag_affinity_dict[user_name][tag] = [(2 * user_tag_diff - MAL_tag_diff) * freq_coeff,
#             #                                           tag_show_list[tag]]
#         except ZeroDivisionError:
#             user_tag_affinity_dict[user_name][tag] = 0


# def get_full_tags_list(shows_tags_dict):
#     # Gets list of all tags + counts amount of show per studio
#     all_tags_list = set()
#     studio_dict={}
#     for show, show_dict in shows_tags_dict.items():
#         if 'Tags' in show_dict:
#             for tag_dict in show_dict['Tags']:
#                 all_tags_list.add(tag_dict['name'])
#         if 'Genres' in show_dict.keys():
#             for genre in show_dict['Genres']:
#                 all_tags_list.add(genre)
#         if 'Studio' in show_dict.keys():
#             if show_dict['Studio']:
#                 if show_dict['Studio'] not in studio_dict:
#                     studio_dict[show_dict['Studio']] = 1
#                 else:
#                     studio_dict[show_dict['Studio']] += 1  # # #
#
#     extra_studios = ['Trigger', 'Passione', 'project No.9', 'POLYGON PICTURES', 'EMT Squared', 'SANZIGEN']
#
#     # Only keeps studios that have over 30 shows or are in extra_studios
#     for studio, amount_of_shows in studio_dict.items():
#         if amount_of_shows >= 30 or studio in extra_studios:
#             all_tags_list.add(studio)
#
#     return all_tags_list


# def get_relevant_show_list():
#     durations = data.anime_df.row(data.anime_db_stats['Duration'], named=True)
#     episodes = data.anime_df.row(data.anime_db_stats['Episodes'], named=True)
#     filtered_cols = [x for x in data.titles
#                      if durations[x] * episodes[x] > 15
#                      and x in shows_tags_dict.keys()
#                      and durations[x] >= 2]
#     return filtered_cols


def get_tag_index(tag_name,show_tags_list):
    for i,tag in enumerate(show_tags_list):
        if tag_name == tag['name']:
            return i
    return -1


def adjust_tag_percentage(tag):
    p = tag['percentage']
    if p >= 85:
        adjusted_p = round((tag['percentage'] / 100), 3)
    elif p >= 70:
        second_coeff = 20 / 3 - tag['percentage'] / 15  # Can be a number from 1 to 1.5
        adjusted_p = round((tag['percentage'] / 100)**second_coeff, 3)
        # 85 ---> 1, 70 ---> 2, 15 forward means -1 so m=-1/15, y-2 = -1/15(x-70) --> y=-1/15x + 20/3
    elif p >= 60:
        second_coeff = 16 - tag['percentage'] / 5
        adjusted_p = round((tag['percentage'] / 100)**second_coeff, 3)
    else:
        adjusted_p = 0
    return adjusted_p


def load_tags_database():
    print(f"Unpacking part 1 of database")
    tag_df = pl.read_parquet(f"{tags_db_filename}-P1.parquet")
    for i in range(2,1000):
        try:
            print(f"Unpacking part {i} of database")
            temp_df = pl.read_parquet(f"{tags_db_filename}-P{i}.parquet")
            tag_df = tag_df.vstack(temp_df)
            # os.remove(f"{tags_db_filename}-P{i}.parquet")
        except FileNotFoundError:
            break
    return tag_df


def create_affinity_DB():

    def calculate_affinities_of_user(user_index):
        # def sum_scores_per_show_tag():
        #     show_tags_list = tags.shows_tags_dict[show]['Tags']
        #     show_genres_list = tags.shows_tags_dict[show]['Genres']
        #     show_studio = tags.shows_tags_dict[show]['Studio']
        #
        #     for tag in show_tags_list:
        #         if tag['name'] not in tags.all_tags_list:
        #             continue
        #         adjusted_p = adjust_tag_percentage(tag)
        #         if adjusted_p == 0:
        #             break
        #
        #         user_score_per_tag[tag['name']] += user_score * adjusted_p
        #         MAL_score_per_tag[tag['name']] += MAL_score * adjusted_p
        #         user_tag_counts[tag['name']] += adjusted_p
        #         tag_show_list[tag['name']].append(show)
        #
        #     for genre in show_genres_list:
        #         user_score_per_tag[genre] += user_score
        #         MAL_score_per_tag[genre] += MAL_score
        #         user_tag_counts[genre] += 1
        #         tag_show_list[genre].append(show)
        #
        #     if show_studio in tags.all_tags_list:
        #         user_score_per_tag[show_studio] += user_score
        #         MAL_score_per_tag[show_studio] += MAL_score
        #         user_tag_counts[show_studio] += 1
        #         tag_show_list[show_studio].append(show)

        nonlocal user_tag_affinity_dict
        nonlocal database_dict

        save_data_per = 10000  # Each batch of N users will be separated into their own mini-database during
        # runtime to avoid blowing up my poor 32GB of RAM
        show_count = 0

        user_name = partial_main_df['Username'][user_index]
        user_scores = partial_main_df.select(relevant_shows).row(user_index, named=True)
        watched_user_score_list = [score for title, score in user_scores.items() if score]
        # ^ Simply a list of the user's scores for every show they watched

        watched_titles = [title for title, score in user_scores.items() if score]
        watched_MAL_score_dict = mean_score_row.select(watched_titles).to_dict(as_series=False)
        watched_MAL_score_list = [score[0] for title, score in
                                  watched_MAL_score_dict.items() if title != "Rows"]
        # ^ Simply a list of the MAL scores for every show the user has watched

        MAL_mean_of_watched = np.mean(watched_MAL_score_list)
        user_mean_of_watched = np.mean(watched_user_score_list)
        user_std_of_watched = np.std(watched_user_score_list)

        user_tag_affinity_dict[user_name] = {tag_name: 0 for tag_name in tags.all_tags_list}
        # ^ A dictionary that will store, for each user, their affinity to each of the 300+ tags we have

        user_tag_counts = {tag_name: 0 for tag_name in tags.all_tags_list}
        # ^ This dictionary will store the WEIGHTED (by each tag's percentage) tag count for each tag.

        MAL_score_per_tag = {tag_name: 0 for tag_name in tags.all_tags_list}
        user_score_per_tag = {tag_name: 0 for tag_name in tags.all_tags_list}
        # ^ The above two are for calculating the average scores per tag later (both user and MAL averages)

        tag_show_list = {tag_name: [] for tag_name in tags.all_tags_list}
        # ^ A dictionary that will store, for each tag, all the shows that have said tag.

        user_show_list = []

        # Now, we will loop over every show that the user has watched
        for show, user_score in user_scores.items():
            if user_scores[show]:

                user_show_list.append(show)
                MAL_score = mean_score_row[show].item()

                if MAL_score < 6.5:
                    continue
                    # We don't want to base decisions on a person's affinity to certain types of shows
                    # on shows that are widely considered bad - for example, a romance fan won't necessarily
                    # like a bad romance show more than someone indifferent towards romance shows.

                show_count += 1
                MAL_score_coeff = (-3 / 8) * MAL_score + 31 / 8

                show_tags_list = tags.shows_tags_dict[show]['Tags']
                show_genres_list = tags.shows_tags_dict[show]['Genres']
                show_studio = tags.shows_tags_dict[show]['Studio']

                for tag in show_tags_list:
                    if tag['name'] not in tags.all_tags_list:
                        continue
                    adjusted_p = adjust_tag_percentage(tag)
                    # On Anilist, every tag a show has is listed with a percentage. This percentage
                    # can be pushed down or up a bit by any registered user, so as a variable it
                    # inevitably introduces human error - a tag with 60% means that not all users agree
                    # that said tag is strongly relevant to the show, and thus we want to reduce its weight
                    # by more than just 40% compared to a 100% tag.
                    if adjusted_p == 0:
                        break

                    user_score_per_tag[tag['name']] += user_score * adjusted_p
                    MAL_score_per_tag[tag['name']] += MAL_score * adjusted_p
                    user_tag_counts[tag['name']] += adjusted_p
                    tag_show_list[tag['name']].append(show)

                for genre in show_genres_list:
                    user_score_per_tag[genre] += user_score
                    MAL_score_per_tag[genre] += MAL_score
                    user_tag_counts[genre] += 1
                    # Genres and studios do not have percentages, so we add 1 as if p=100
                    tag_show_list[genre].append(show)

                if show_studio in tags.all_tags_list:
                    user_score_per_tag[show_studio] += user_score
                    MAL_score_per_tag[show_studio] += MAL_score
                    user_tag_counts[show_studio] += 1
                    tag_show_list[show_studio].append(show)


        # After calculating the total scores for each tag, we calculate the "affinity" of the user
        # to each tag.
        for tag in tags.all_tags_list:
            try:
                tag_overall_ratio = user_tag_counts[tag] / show_count
                freq_coeff = min(1, max(user_tag_counts[tag] / 10, tag_overall_ratio * 20))
                # User has to watch either at least 10 shows with the tag or have the tag in at least 5% of their watched
                # shows for it to count fully.
                user_tag_diff = user_score_per_tag[tag] / user_tag_counts[tag] - user_mean_of_watched
                MAL_tag_diff = MAL_score_per_tag[tag] / user_tag_counts[tag] - MAL_mean_of_watched
                user_tag_affinity_dict[user_name][tag] = (2*user_tag_diff - MAL_tag_diff) * freq_coeff

            except ZeroDivisionError:
                user_tag_affinity_dict[user_name][tag] = 0

        user_tag_affinity_dict[user_name]['Shows'] = user_show_list

        # Now we choose 20 random shows from the user's list. Each show will give us an entry in the final
        # database. Each entry will consist of the user's affinities to each genre, their
        random.shuffle(user_show_list)
        user_shows_sample = user_show_list[0:20]

        for show in user_shows_sample:
            show_tags_list = tags.shows_tags_dict[show]['Tags']
            show_genres_list = tags.shows_tags_dict[show]['Genres']
            show_studio = tags.shows_tags_dict[show]['Studio']


            # On Anilist, every show has other shows that are recommended by users. The "rating"
            # is how many users recommended it. We want to create a "Recommended Shows Affinity" column
            # while taking the weighted average of the user's affinities to the recommended shows.
            # The weight of each show is it's rating/total rating of all recommended shows.

            total_rec_rating = sum([rating for title,rating in tags.shows_tags_dict[show]['Recommended'].items()])
            recommended_shows = tags.shows_tags_dict[show]['Recommended']
            rec_affinity = 0

            for rec_anime, rec_rating in recommended_shows.items():
                if rec_anime in relevant_shows and user_scores[rec_anime] and rec_rating > 0:
                    MAL_score = mean_score_row[rec_anime].item()
                    MAL_score_coeff = (-3 / 8) * MAL_score + 31 / 8
                    user_diff = user_scores[rec_anime] - user_mean_of_watched
                    MAL_diff = watched_MAL_score_dict[rec_anime][0] - MAL_mean_of_watched
                    rec_affinity += (user_diff - MAL_diff*MAL_score_coeff)*rec_rating

            try:
                weighted_aff = rec_affinity/total_rec_rating
                if np.isnan(weighted_aff) or np.isinf(weighted_aff):
                    raise ValueError
                database_dict['Recommended Shows Affinity'].append(weighted_aff)
            except (ZeroDivisionError, ValueError) as e: # No relevant recommended shows
                database_dict['Recommended Shows Affinity'].append(0)

            for tag in tags.all_tags_list:
                index = get_tag_index(tag, show_tags_list)
                if index!=-1:
                    adjusted_p = adjust_tag_percentage(show_tags_list[index])
                    database_dict[tag].append(adjusted_p)
                else:
                    if tag not in show_genres_list and tag!= show_studio:
                        database_dict[tag].append(0)
                database_dict[f"{tag} Affinity"].append(user_tag_affinity_dict[user_name][tag])

            for genre in show_genres_list:
                database_dict[genre].append(1)

            if show_studio in tags.all_tags_list:
                database_dict[show_studio].append(1)

            database_dict['User Score'].append(user_scores[show])
            database_dict['Show Score'].append(mean_score_row[show].item())
            database_dict['Mean Score'].append(user_mean_of_watched)
            database_dict['Standard Deviation'].append(user_std_of_watched)

        if (user_index+1) % save_data_per==0:
            print(f"Finished processing user {user_index+1}")
            save_pickled_file(f"user_tag_affinity_dict-P{(user_index+1) // save_data_per}.pickle",
                              user_tag_affinity_dict)
            # Fix the above (either remove this entirely or concatenate them at the end?)

            for key in database_dict.keys():
                database_dict[key] = np.array(database_dict[key], dtype=np.float32)

            pl.DataFrame(database_dict).write_parquet(f"{tags_db_filename}-P{(user_index + 1) // save_data_per}.parquet")

            # After saving the data, we need to reinitialize the dicts to avoid wasting memory
            for tag in tags.all_tags_list:
                database_dict[tag] = []
                database_dict[f"{tag} Affinity"] = []

            database_dict = database_dict | {"Recommended Shows Affinity": [],
                    "Show Score": [], "Mean Score" : [], "Standard Deviation" : [], "User Score": []}
            user_tag_affinity_dict = {}

    data = Data()
    tags = Tags()
    relevant_shows = list(tags.shows_tags_dict.keys())
    # The shows in our tags dict are the ones filtered when creating it
    # ( >= 15 min in length, >= 2 min per ep)

    partial_main_df = data.main_df.select(data.main_db_stats + relevant_shows)
    partial_anime_df = data.anime_df.select(["Rows"] + relevant_shows)

    mean_score_row = partial_anime_df.filter(pl.col('Rows') == "Mean Score")
    user_amount = partial_main_df.shape[0]

    database_dict={}
    for tag in tags.all_tags_list:
        database_dict[tag] = []
        database_dict[f"{tag} Affinity"] = []
    database_dict = database_dict | {"Recommended Shows Affinity": [],
                    "Show Score": [], "Mean Score": [], "Standard Deviation" : [], "User Score": []}

    # try:
    #     # user_tag_affinity_dict = load_pickled_file("user_tag_affinity_dict3.pickle")
    #     # #Change this
    #     tags_db = pl.read_parquet("UserTagsDB.parquet")

    # except FileNotFoundError:
    for user_index in range(user_amount):
        if user_index%100==0:
            print(f"Currently on user {user_index}")
        user_tag_affinity_dict = {}
        calculate_affinities_of_user(user_index)


    tags_db = load_tags_database()
    remove_zero_columns(tags_db)
    tags_db.write_parquet(f"{tags_db_filename}.parquet")


if __name__ == '__main__':
    data = Data()
    create_affinity_DB()
    # tags_db = load_tags_database()
    # remove_zero_columns(tags_db)
    # tags_db.write_parquet(tags_db_filename)
    print(5)










