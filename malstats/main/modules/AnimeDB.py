import datetime
import logging
import shutil

from django.db import transaction
from polars import ColumnNotFoundError
from main.modules.filenames import *
from main.modules.MAL_utils import Seasons, MediaTypes, MALUtils
from main.modules.general_utils import add_suffix_to_filename
from main.modules.GlobalValues import MINIMUM_SCORE
import django
import os
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'animisc.settings')
django.setup()
from main.models import AnimeData, AnimeDataUpdated

logger = logging.getLogger('nyanpasutats')


class AnimeDB:
    """This class creates AnimeDB.parquet and holds all relevant functions and information
    regarding it. Current fields :

    - ID
    - Mean Score
    - Scores (how many users scored the show)
    - Members (how many users have the show in their list)
    - Episodes
    - Duration (of each episode)
    - Type (see MediaTypes)
    - Year
    - Season

    Note : The above-mentioned fields are rows, not columns, for easier synchronization with UserDB.

    """

    # ┌──────┬────────────┬────────────┬───────────┬───┬──────────┬────────────┬────────────┬────────────┐
    # │ Rows ┆ Fullmetal  ┆ Bleach:    ┆ Steins;Ga ┆ … ┆ Kokuhaku ┆ Hametsuno  ┆ Utsu       ┆ Tenkuu     │
    # │ ---  ┆ Alchemist: ┆ Sennen     ┆ te        ┆   ┆ ---      ┆ Mars       ┆ Musume     ┆ DanzaiSke  │
    # │ str  ┆ Brotherhoo ┆ Kessen -hen┆ ---       ┆   ┆ f64      ┆ ---        ┆ Sayuri     ┆ lter + Heav│
    # │      ┆ d          ┆ ---        ┆ f64       ┆   ┆          ┆ f64        ┆ ---        ┆ en         │
    # │      ┆ ---        ┆ f64        ┆           ┆   ┆          ┆            ┆ f64        ┆ ---        │
    # │      ┆ f64        ┆            ┆           ┆   ┆          ┆            ┆            ┆ f64        │
    # ╞══════╪════════════╪════════════╪═══════════╪═══╪══════════╪════════════╪════════════╪════════════╡
    # │ ID   ┆ 5114.0     ┆ 41467.0    ┆ 9253.0    ┆ … ┆ 31634.0  ┆ 413.0      ┆ 13405.0    ┆ 3287.0     │
    # │ Mean ┆ 9.1        ┆ 9.07       ┆ 9.07      ┆ … ┆ 2.3      ┆ 2.22       ┆ 1.98       ┆ 1.85       │
    # │ Scor ┆            ┆            ┆           ┆   ┆          ┆            ┆            ┆            │
    # │ e    ┆            ┆            ┆           ┆   ┆          ┆            ┆            ┆            │
    # │ Scor ┆ 2.02003e6  ┆ 214073.0   ┆ 1.336233e ┆ … ┆ 4906.0   ┆ 47641.0    ┆ 15878.0    ┆ 26435.0    │
    # │ es   ┆            ┆            ┆ 6         ┆   ┆          ┆            ┆            ┆            │
    # │ Memb ┆ 3.176593e6 ┆ 445356.0   ┆ 2.44046e6 ┆ … ┆ 6699.0   ┆ 65639.0    ┆ 20794.0    ┆ 37333.0    │
    # │ ers  ┆            ┆            ┆           ┆   ┆          ┆            ┆            ┆            │
    # │ Epis ┆ 64.0       ┆ 13.0       ┆ 24.0      ┆ … ┆ 1.0      ┆ 1.0        ┆ 1.0        ┆ 1.0        │
    # │ odes ┆            ┆            ┆           ┆   ┆          ┆            ┆            ┆            │
    # │ Dura ┆ 24.3       ┆ 24.5       ┆ 24.3      ┆ … ┆ 0.7      ┆ 19.7       ┆ 3.4        ┆ 19.0       │
    # │ tion ┆            ┆            ┆           ┆   ┆          ┆            ┆            ┆            │
    # │ Type ┆ 1.0        ┆ 1.0        ┆ 1.0       ┆ … ┆ 5.0      ┆ 3.0        ┆ 3.0        ┆ 3.0        │
    # │ Year ┆ 2009.0     ┆ 2022.0     ┆ 2011.0    ┆ … ┆ 2015.0   ┆ 2005.0     ┆ 2003.0     ┆ 2004.0     │
    # │ Seas ┆ 2.0        ┆ 4.0        ┆ 2.0       ┆ … ┆ 3.0      ┆ 3.0        ┆ 1.0        ┆ 4.0        │
    # │ on   ┆            ┆            ┆           ┆   ┆          ┆            ┆            ┆            │
    # └──────┴────────────┴────────────┴───────────┴───┴──────────┴────────────┴────────────┴────────────┘

    _instances = {}
    # noinspection DuplicatedCode

    def __new__(cls, filename=None, reset=False, *args, **kwargs):
        """The class is a Multiton, currently used for two different databases - the original
        one created at the start of every data gathering cycle (AnimeDB.parquet),
        and all of its subsequent updates (AnimeDB-U.parquet, updated daily)"""
        if filename not in cls._instances or reset:
            instance = super().__new__(cls)
            cls._instances[filename] = instance

            instance._filename = anime_database_name if not filename else filename
            instance._df = None
            instance._sql_db = AnimeData if not filename or filename == anime_database_name else AnimeDataUpdated
            instance._titles = None
            instance._partial_df = None
            instance._df_metadata = None
            instance._ids = None
            instance._id_title_map = None
            instance._mean_scores = None
            instance._scored_amounts = None
            instance._members = None
            instance._episodes = None
            instance._durations = None
            instance._media_types = None
            instance._seasons = None
            instance._years = None
        return cls._instances[filename]

    @classmethod
    def reset(cls):
        cls._instances = {}

    stats = {'ID': 0, 'Mean Score': 1, 'Scores': 2, 'Members': 3, 'Episodes': 4,
             'Duration': 5, 'Type': 6, 'Year': 7, 'Season': 8}

    @property
    def df(self):
        if not isinstance(self._df, pl.DataFrame):
            try:
                print("Loading anime database")
                self._df = pl.read_parquet(self._filename)
                print("Anime database loaded successfully")
            except FileNotFoundError:
                print("Anime database not found. Creating new anime database")
                if "-P" in self._filename.name:
                    self.create_partial_df()
                else:
                    self.generate_anime_DB()
                self._df = pl.read_parquet(self._filename)
        return self._df

    @property
    def df_metadata(self):
        if not self._df_metadata:
            try:
                self._df_metadata = pq.ParquetFile(self._filename)
            except FileNotFoundError:
                self.generate_anime_DB()
                self._df_metadata = pq.ParquetFile(self._filename)
        return self._df_metadata

    @property
    def titles(self):  # change this into a normal var?
        if not self._titles:
            columns = [field.name for field in self.df_metadata.schema]
            self._titles = columns[1:]
        return self._titles

    def create_partial_df(self):
        full_db = AnimeDB(anime_database_updated_name if "-U" in self._filename.name else anime_database_name)
        df_dict = full_db.df.to_dict(as_series=False)
        titles = [title for title, show_stats in df_dict.items()
                  if (title != 'Rows' and
                  self.show_meets_standard_conditions_nls(show_stats))]
        partial_filename = add_suffix_to_filename(self._filename, "P")
        self._partial_df = full_db.df.select(["Rows"] + titles)
        self._partial_df.write_parquet(partial_filename)

    @property
    def partial_df(self):
        if not self._partial_df:
            partial_filename = add_suffix_to_filename(self.filename, "P")
            try:
                self._partial_df = pl.read_parquet(partial_filename)
            except FileNotFoundError:
                self._partial_df = self.create_partial_df()
                self._partial_df.write_parquet(partial_filename)
        return self._partial_df

    def filter_titles(self, meets_conditions_func, title_list=None):
        df_dict = self.df.to_dict(as_series=False)
        titles = [title for title, show_stats in df_dict.items()
                  if title != 'Rows'
                  and (title_list is None or title in title_list)
                  and meets_conditions_func(show_stats)
                  ]
        return titles

    def sort_titles_by_release_date(self, titles):
        anime_db = self.df.select(titles)
        df_dict = anime_db.to_dict(as_series=False)
        stats = {title: {'ID': show_stats[self.stats["ID"]],
                         'Year': show_stats[self.stats["Year"]],
                         'Season': show_stats[self.stats["Season"]]
                         } for title, show_stats in df_dict.items()}

        titles = [title for title, stats in sorted(stats.items(),
                  key=lambda x: (x[1]['Year'], x[1]['Season']))]

        return titles

    @property
    def ids(self):
        if not self._ids:
            ids_row = self.df.row(self.stats['ID'])
            self._ids = {title: ID for (title, ID)
                         in list(zip(self.titles, ids_row[1:]))}
        return self._ids

    @property
    def id_title_map(self):
        if not self._id_title_map:
            self._id_title_map = {ID: title for (title, ID) in self.ids.items()}
        return self._id_title_map

    @property
    def mean_scores(self):
        if not self._mean_scores:
            mean_score_row = self.df.row(self.stats['Mean Score'])
            self._mean_scores = {title: mean_score for (title, mean_score)
                                 in list(zip(self.titles, mean_score_row[1:]))}
        return self._mean_scores

    @property
    def scored_amounts(self):
        if not self._scored_amounts:
            scored_amounts_row = self.df.row(self.stats['Scores'])
            self._scored_amounts = {title: scored_amount for (title, scored_amount)
                                    in list(zip(self.titles, scored_amounts_row[1:]))}
        return self._scored_amounts

    @property
    def members(self):
        if not self._members:
            members_row = self.df.row(self.stats['Members'])
            self._members = {title: member_count for (title, member_count)
                             in list(zip(self.titles, members_row[1:]))}
        return self._members

    @property
    def episodes(self):
        if not self._episodes:
            episodes_row = self.df.row(self.stats['Episodes'])
            self._episodes = {title: episode_count for (title, episode_count)
                              in list(zip(self.titles, episodes_row[1:]))}
        return self._episodes

    @property
    def durations(self):
        if not self._durations:
            durations_row = self.df.row(self.stats['Duration'])
            self._durations = {title: duration for (title, duration)
                               in list(zip(self.titles, durations_row[1:]))}
        return self._durations

    @property
    def media_types(self):
        if not self._media_types:
            media_types_row = self.df.row(self.stats['Type'])
            self._media_types = {title: media_type for (title, media_type)
                                 in list(zip(self.titles, media_types_row[1:]))}
        return self._media_types

    @property
    def seasons(self):
        if not self._seasons:
            seasons_row = self.df.row(self.stats['Season'])
            self._seasons = {title: season for (title, season) in list(zip(self.titles, seasons_row[1:]))}
        return self._seasons

    @property
    def converted_seasons(self):
        return {title: Seasons(season_num).name if season_num else None for (title, season_num) in self.seasons.items()}

    @property
    def years(self):
        if not self._years:
            years_row = self.df.row(self.stats['Year'])
            self._years = {title: year for (title, year) in list(zip(self.titles, years_row[1:]))}
        return self._years

    @classmethod
    def show_is_from_current_season(cls, show_stats: dict):
        if show_stats[cls.stats["Season"]] == MALUtils.get_current_season()['Season'].value\
                and show_stats[cls.stats["Year"]] == datetime.datetime.now().year:
            return True
        return False

    @classmethod
    def show_meets_standard_conditions(cls, show_stats: dict):
        if (int((show_stats[cls.stats["Scores"]])) >= 2000
                or cls.show_is_from_current_season(show_stats))  \
                and show_stats[cls.stats["Duration"]] * \
                show_stats[cls.stats["Episodes"]] >= 15\
                and show_stats[cls.stats["Duration"]] >= 3:
                # and show_stats[self.stats["Year"]]>=2021 \
                # and show_stats[self.stats["Year"]]<=2022:
                # Remember to change condition_func inside
                # initialize_graph_collection as well until
                # refactoring
            return True
        return False

    @classmethod
    def show_meets_standard_conditions_nls(cls, show_stats: dict):
        if (int(show_stats[cls.stats["Scores"]]) >= 2000
                or cls.show_is_from_current_season(show_stats)) \
                and show_stats[cls.stats["Duration"]] * \
                show_stats[cls.stats["Episodes"]] >= 15 \
                and show_stats[cls.stats["Duration"]] >= 3 \
                and show_stats[cls.stats["Mean Score"]] >= MINIMUM_SCORE:
            return True
        return False

    def get_id_by_title(self, title):
        return self.df[title][self.stats['ID']]

    def get_stats_of_shows(self, show_list, relevant_stats: list):
        """ Will create a dictionary that has every show in show_list as the key, and every stat in relevant_stats
            in a list as the value.
            Example :
            {'Shingeki no Kyojin': {'ID': 16498.0, 'Mean Score': 8.53, 'Members': 3717089.0},
             'Shingeki no Kyojin OVA': {'ID': 18397.0, 'Mean Score': 7.87, 'Members': 439454.0}"""

        stats_dict = {}
        for show in show_list:
            show_dict = {}
            for stat in relevant_stats:
                try:
                    # print("Filename before error is", self._filename)
                    show_dict[stat] = self.df.filter(pl.col('Rows') == stat)[show].item()
                except ColumnNotFoundError:
                    break
                except ValueError:
                    show_dict[stat] = None
            if show_dict:
                stats_dict[show] = show_dict
        return stats_dict

    def generate_anime_DB(self, non_sequels_only=False, update=False):

        def create_anime_DB_entry(anime):
            # Helper function, creates a list which will later serve as a column in the anime DB.

            def parse_start_date(date_str):
                try:
                    date_parsed = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    return date_parsed.year, (date_parsed.month - 1) // 3 + 1
                except ValueError:
                    try:
                        date_parsed = datetime.datetime.strptime(date_str, "%Y-%m")
                        return date_parsed.year, (date_parsed.month - 1) // 3 + 1
                    except ValueError:
                        return int(date_str), 1  # Only the year is known, season is assumed to be Winter (placeholder)

            anime_data = []
            for field in fields_no_edit_needed:  # This takes care of the fields that can be put into db as they are
                try:
                    anime_data.append(anime["node"][field])
                except KeyError:
                    return  # Critical fields, ignore entry if they're missing

            # The rest of the fields need editing or error handling
            if anime["node"]["num_episodes"]:
                anime_data.append(anime["node"]["num_episodes"])
            else:  # This means episode_num = null, can only happen with VERY obscure shows.
                anime_data.append(1)

            if anime["node"]["average_episode_duration"]:
                anime_data.append(round(anime["node"]["average_episode_duration"] / 60, 1))
            else:  # Same as above
                anime_data.append(20)

            try:
                media_type_index = MediaTypes[anime['node']["media_type"]].value
            except KeyError:
                media_type_index = None

            anime_data.append(media_type_index)

            try:
                year = int(anime["node"]["start_season"]["year"])
                season = anime["node"]["start_season"]["season"]
                season = Seasons[season].value  # We convert the season to a numerical value for the database
            except KeyError:
                try:
                    year, season = parse_start_date(anime["node"]["start_date"])
                except KeyError:
                    # year, season = None, None
                    return  # Critical fields, ignore entry if they're missing
            anime_data.append(year)

            correct_seasons = {'Sousou no Frieren': 4}
            # Stupid API has Frieren as a summer anime,
            # it'll probably have more mistakes in that vein

            title = anime["node"]["title"]
            if title in correct_seasons.keys():
                anime_data.append(correct_seasons[title])
            else:
                anime_data.append(season)

            anime_data_dict[title] = anime_data

            image_url = anime['node'].get('main_picture', {}).get('medium', "")

            anime_data_for_db = anime_data + [title, image_url]
            db_fields = ['mal_id', 'mean_score', 'scores', 'members', 'episodes', 'duration', 'type', 'year', 'season',
                         'name', 'image_url']

            return dict(zip(db_fields, anime_data_for_db))

        last_show_reached = False
        last_show = 'Tenkuu Danzai'
        # Lowest rated show, it'll be the last one in the sorted list.
        # Hardcoded because it's extremely unlikely to be dethroned from its
        # lowest rated show ever title.

        url_required_fields = ["id", "mean", "num_scoring_users", "num_list_users", "num_episodes",
                               "average_episode_duration",
                               "media_type", "start_season", "start_date"]

        fields_no_edit_needed = ["id", "mean", "num_scoring_users", "num_list_users"]
        # The fields we need from the JSON object containing information about a single anime.

        stat_names = list(self.stats.keys())
        anime_data_dict = {'Rows': stat_names}

        anime_entries = []

        page_num = 0
        while not last_show_reached:
            # We loop over pages of anime info that we get from the Jikan API (the info is
            # sorted by score, from highest to lowest) until we reach the lowest rated show.
            # There should be 100 shows per page/batch.

            anime_batch = MALUtils.get_anime_batch_from_MAL(page_num, url_required_fields)
            try:
                print(f"Currently on score {anime_batch['data'][-1]['node']['mean']}")
                logger.info(f"Currently on score {anime_batch['data'][-1]['node']['mean']}")
            except KeyError:
                print("Finished gathering data. Saving anime database")
                logger.info("Finished gathering data. Saving anime database")

            for anime in anime_batch["data"]:
                if not non_sequels_only:

                    anime_entry = create_anime_DB_entry(anime)
                    if not anime_entry:
                        continue

                    anime_entries.append(anime_entry)
                    title = anime_entry['name']
                if title.startswith(last_show):
                    last_show_reached = True
                    break
            page_num += 1

        table = pa.Table.from_pydict(anime_data_dict)
        if os.path.exists(anime_database_updated_name):
            # If this isn't the first update, we save the previous update as well,
            # since we'll need it for updating graphs, tags etc
            shutil.copy(anime_database_updated_name, anime_database_prev_updated_name)
        pq.write_table(table, self._filename)  # This creates a .parquet file from the dict

        if self._filename == anime_database_name:
            # When we create the db for the first time, updated is the same as original
            # and prev_updated is same as updated
            shutil.copy(anime_database_name, anime_database_updated_name)
            shutil.copy(anime_database_updated_name, anime_database_prev_updated_name)
        logger.info("Anime database successfully created")

        # Update the SQL database
        with transaction.atomic():
            for entry in anime_entries:
                self._sql_db.objects.update_or_create(
                    mal_id=entry['mal_id'],
                    defaults=entry
                )

    def are_separate_shows(self, show1: str, show2: str, relation_type: str):
        """ This method tries to determine whether two entries that are related in some way on MAL
        are the same show, or two different shows.

        Note : The methodology used here is very rough, and relies purely on how the shows are related
        to each other, their media type, and their length. There are two reasons for this :

        1) Even the definition of same show vs different shows is very rough - for example,
        is Fate/Zero the same show as Fate/Unlimited Blade Works? Is A Certain Magical Index
        the same show as A Certain Scientific Railgun?

        2) Even if you count Fate/Zero and Fate/UBW as different shows, there is literally no way
        to separate them going purely by their MAL entries. Fate/UBW is listed as a sequel of Fate/Zero,
        both are full-length TV shows, and both have a very sizeable watcher amount. There are multiple cases
        of shows like these where it's simply impossible to separate them due to how MAL classifies them
        (sometimes outright misclassifies, like putting alternative_version (which should NOT count as a
        separate show, since alternative version is basically the same show but made in a different time/
        from a different perspective) instead of alternative_setting (which usually means same universe but
        completely different characters, and would almost always be a different show).

        In short, we can only rely on non-fully-accurate MAL data to separate what would be difficult
        even for humans to agree on, so this won't be 100% precise.

         """

        # def both_shows_are_TV():
        #     return show_stats[show1]["Type"] == 1 and show_stats[show2]["Type"] == 1
        #
        # def both_shows_are_movies():
        #     return show_stats[show1]["Type"] == 2 and show_stats[show2]["Type"] == 2

        def show_is_longer_than(minutes, name):
            if not show_stats[name]["Episodes"]:
                show_stats[name]["Episodes"] = 1
            return show_stats[name]["Duration"] * show_stats[name]["Episodes"] > minutes

        if show1 not in self.titles or show2 not in self.titles:  # take care of this outside later
            return False
        if relation_type in ['sequel', 'prequel', 'summary']:
            # Sequels, prequels, alternative versions and summaries are never separate shows
            return False

        if relation_type == 'character':
            # "character" means that the only common thing between the two shows is that some of
            # the characters are mutual. It will always be a separate show, or something very short
            # that isn't in the partial database in the first place.
            return True

        relevant_stats = ["Duration", "Episodes", "Type"]
        show_stats = self.get_stats_of_shows([show1, show2], relevant_stats)

        if relation_type in ['other', 'side_story', 'alternate_version', 'spin_off', 'parent_story']:
            # This is the most problematic case. MAL is very inconsistent with how it labels things
            # as "other", "side story" or "spin-off". The latter two are used almost interchangeably,
            # and "other" can be used for pretty much ANYTHING. Side stories/spin-offs, commercials,
            # even crossovers. There is no feasible way to catch literally every case, but this gets
            # the vast majority of them.
            if show_is_longer_than(150, show1) and show_is_longer_than(150, show2):
                return True  # Add a search for what sequels are?
            return False

        if relation_type == 'alternative_setting':
            # Alternative setting almost always means that the shows are set in the same universe,
            # but have different stories or even characters. Sometimes it can also be used for
            # miscellanous related shorts, which is why I made a small (arbitrary) length requirement
            # for the shows to be counted as separate.

            if (show_is_longer_than(60, show1) and show_is_longer_than(60, show2)):
                return True
            return False

        return False

    @staticmethod
    def get_post_update_changed_titles(update_from_scratch=False):
        current_updated_db = pl.read_parquet(anime_database_updated_name)

        previous_updated_db = pl.read_parquet(anime_database_prev_updated_name
                                              if not update_from_scratch else anime_database_name)
        print(f"Length of updated filename : {len(current_updated_db.columns)}")
        print(update_from_scratch)
        print(f"Length of updated filename : {len(previous_updated_db.columns)}")
        new_titles = list(set(current_updated_db.columns) - set(previous_updated_db.columns))
        changed_titles = list(set(previous_updated_db.columns) - set(current_updated_db.columns))
        logger.info(f"Updated AnimeDB. \n New titles : {new_titles} \n Changed/Removed titles : {changed_titles}")
        return new_titles, changed_titles

    # def get_title_by_id(self, malId):
    #     try:
    #         index = self.anime_db.ids.index(malId)
    #     except ValueError:
    #         return
    #
    #     try:
    #         MAL_title = self.anime_db.titles[index]
    #         return MAL_title
    #     except IndexError:
    #         return




