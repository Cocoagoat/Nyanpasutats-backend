from AnimeDB import AnimeDB
from UserDB import UserDB
from Tags import Tags
from filenames import *
from MAL_utils import *
from Graphs import Graphs


class AffinityDB:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # All properties are loaded on demand
        self._df = None
        self.anime_df = AnimeDB()
        self.tags = Tags()
        self.graphs = Graphs()

    @property
    def df(self):
        """Polars database (stored as .parquet), contains username
        + mean score + scored shows + all the scores of each user in it."""
        if not isinstance(self._df, pl.DataFrame):
            try:
                print("Loading affinity database")
                self._df = pl.read_parquet(f"{tags_db_filename}.parquet")
            except FileNotFoundError:
                print("Affinity database not found. Creating new affinity database")
                self.create_affinity_DB()
                self._df = pl.read_parquet(f"{tags_db_filename}.parquet")
        return self._df

    def create_affinity_DB(self):

        def calculate_affinities_of_user(user_index):

            nonlocal user_tag_affinity_dict
            nonlocal database_dict

            save_data_per = 10  # Each batch of N users will be separated into their own mini-database during
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
            processed_shows = []

            # Now, we will loop over every show that the user has watched
            for show, user_score in user_scores.items():

                if show in processed_shows:
                    continue

                show_related_entries = self.graphs.find_related_entries(show)
                stats_of_related_entries = anime_db.get_stats_of_shows(show_related_entries,
                                                                       ["Episodes", "Duration"])

                # all_entries_tags_list = list(set([tag for entry in show_related_entries
                #                              for tag in tags.shows_tags_dict[entry]['Tags']]))

                all_entries_tags_list = [{tag['name']: tag['percentage']} for entry in show_related_entries
                                         for tag in self.tags.shows_tags_dict[entry]['Tags']]

                for tag in all_entries_tags_list:
                    max_tag_value = max()
                for entry in show_related_entries:
                    length_coeff = min(1, stats_of_related_entries[show]["Episodes"] *
                                       stats_of_related_entries[show]["Duration"]/200)
                    show_count += length_coeff


                    # for tag in entry_tags_list:
                    #     if tag['name'] not in tags.all_tags_list:
                    #         continue
                    #     adjusted_p = tags.adjust_tag_percentage(tag)
                    #     if adjusted_p == 0:
                    #         break
                # user_score_per_tag[tag['name']] += user_score * adjusted_p
                # MAL_score_per_tag[tag['name']] += MAL_score * adjusted_p
                # user_tag_counts[tag['name']] += adjusted_p
                # tag_show_list[tag['name']].append(show)

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
                        adjusted_p = tags.adjust_tag_percentage(tag)
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

            # The above loop = func1? First put everything in classes tho

            # After calculating the total scores for each tag, we calculate the "affinity" of the user
            # to each tag.
            for tag in tags.all_tags_list:
                try:
                    tag_overall_ratio = user_tag_counts[tag] / show_count
                    freq_coeff = min(1, max(user_tag_counts[tag] / 10, tag_overall_ratio * 20))
                    # User has to watch either at least 10 shows with the tag or have the tag
                    # in at least 5% of their watched shows for it to count fully.
                    user_tag_diff = user_score_per_tag[tag] / user_tag_counts[tag] - user_mean_of_watched
                    MAL_tag_diff = MAL_score_per_tag[tag] / user_tag_counts[tag] - MAL_mean_of_watched
                    user_tag_affinity_dict[user_name][tag] = (2 * user_tag_diff - MAL_tag_diff) * freq_coeff

                except ZeroDivisionError:
                    user_tag_affinity_dict[user_name][tag] = 0

            user_tag_affinity_dict[user_name]['Shows'] = user_show_list

            # Whatever is above - func2 (create_user_tag_affinity_dict? also draw it in the comments)
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

                total_rec_rating = sum([rating for title, rating in tags.shows_tags_dict[show]['Recommended'].items()])
                recommended_shows = tags.shows_tags_dict[show]['Recommended']
                rec_affinity = 0

                for rec_anime, rec_rating in recommended_shows.items():
                    if rec_anime in relevant_shows and user_scores[rec_anime] and rec_rating > 0:
                        MAL_score = mean_score_row[rec_anime].item()
                        MAL_score_coeff = (-3 / 8) * MAL_score + 31 / 8
                        user_diff = user_scores[rec_anime] - user_mean_of_watched
                        MAL_diff = watched_MAL_score_dict[rec_anime][0] - MAL_mean_of_watched
                        rec_affinity += (user_diff - MAL_diff * MAL_score_coeff) * rec_rating

                try:
                    weighted_aff = rec_affinity / total_rec_rating
                    if np.isnan(weighted_aff) or np.isinf(weighted_aff):
                        raise ValueError
                    database_dict['Recommended Shows Affinity'].append(weighted_aff)
                except (ZeroDivisionError, ValueError) as e:  # No relevant recommended shows
                    database_dict['Recommended Shows Affinity'].append(0)

                for tag in tags.all_tags_list:
                    index = tags.get_tag_index(tag, show_tags_list)
                    if index != -1:
                        adjusted_p = tags.adjust_tag_percentage(show_tags_list[index])
                        database_dict[tag].append(adjusted_p)
                    else:
                        if tag not in show_genres_list and tag != show_studio:
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

            if (user_index + 1) % save_data_per == 0:
                print(f"Finished processing user {user_index + 1}")
                save_pickled_file(f"user_tag_affinity_dict-P{(user_index + 1) // save_data_per}.pickle",
                                  user_tag_affinity_dict)
                # Fix the above (either remove this entirely or concatenate them at the end?)

                for key in database_dict.keys():
                    database_dict[key] = np.array(database_dict[key], dtype=np.float32)

                pl.DataFrame(database_dict).write_parquet(
                    f"{tags_db_filename}-P{(user_index + 1) // save_data_per}.parquet")

                # After saving the data, we need to reinitialize the dicts to avoid wasting memory
                for tag in tags.all_tags_list:
                    database_dict[tag] = []
                    database_dict[f"{tag} Affinity"] = []

                database_dict = database_dict | {"Recommended Shows Affinity": [],
                                                 "Show Score": [], "Mean Score": [], "Standard Deviation": [],
                                                 "User Score": []}
                user_tag_affinity_dict = {}

        user_db = UserDB()
        anime_db = AnimeDB()
        tags = Tags()
        graphs = Graphs()
        relevant_shows = list(tags.shows_tags_dict.keys())
        # The shows in our tags dict are the ones filtered when creating it
        # ( >= 15 min in length, >= 2 min per ep)

        partial_main_df = user_db.df.select(user_db.stats + relevant_shows)
        partial_anime_df = anime_db.df.select(["Rows"] + relevant_shows)

        mean_score_row = partial_anime_df.filter(pl.col('Rows') == "Mean Score")
        user_amount = partial_main_df.shape[0]


        database_dict = {}
        for tag in tags.all_tags_list:
            database_dict[tag] = []
            database_dict[f"{tag} Affinity"] = []
        database_dict = database_dict | {"Recommended Shows Affinity": [],
                                         "Show Score": [], "Mean Score": [], "Standard Deviation": [], "User Score": []}

        for user_index in range(user_amount):
            if user_index % 100 == 0:
                print(f"Currently on user {user_index}")
            user_tag_affinity_dict = {}
            calculate_affinities_of_user(user_index)

        tags_db = tags.load_tags_database()
        remove_zero_columns(tags_db)
        tags_db.write_parquet(f"{tags_db_filename}.parquet")
