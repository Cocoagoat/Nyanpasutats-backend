from filenames import *
from MAL_utils import *


class AnimeDB:
    _instance = None

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

    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # All properties are loaded on demand
        self._df = None
        self._titles = None

    stats = {'ID': 0, 'Mean Score': 1, 'Scores': 2, 'Members': 3, 'Episodes': 4,
             'Duration': 5, 'Type': 6, 'Year': 7, 'Season': 8}

    @property
    def df(self):
        """Database with info on all the anime that's listed (and ranked) on MAL.
        Fields are as shown in the "required_fields" variable of generate_anime_DB."""
        if not isinstance(self._df, pl.DataFrame):
            try:
                print("Loading anime database")
                self._df = pl.read_parquet(anime_database_name)
                print("Anime database loaded successfully")
            except FileNotFoundError:
                print("Anime database not found. Creating new anime database")
                self.generate_anime_DB()
                self._df = pl.read_parquet(anime_database_name)
        return self._df

    @property
    def titles(self):  # change this into a normal var?
        """A list of all the anime titles."""
        if not self._titles:
            self._titles = self.df.columns[1:]  # anime_df will automatically be generated
        return self._titles

    def get_stats_of_shows(self, show_list, relevant_stats):
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
                    show_dict[stat] = self.df.filter(pl.col('Rows') == stat)[show].item()
                except ColumnNotFoundError:
                    break
                except ValueError:
                    show_dict[stat] = None
            if show_dict:
                stats_dict[show] = show_dict
        return stats_dict

    def generate_anime_DB(self, non_sequels_only=False):
        """Creates the anime database which contains data (as defined in required_fields)
        on every show"""

        def create_anime_DB_entry(anime, required_fields):
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
                anime_data.append(anime["node"][field])

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
                    year, season = None, None
            anime_data.append(year)
            anime_data.append(season)

            title = anime["node"]["title"]
            anime_data_dict[title] = anime_data
            return title  # Title is returned to check whether we reached the last show

        last_show_reached = False
        last_show = 'Tenkuu Danzai'
        # Lowest rated show, it'll be the last one in the sorted list.
        # Hardcoded because I'm 100% sure nothing will ever be rated
        # lower than THAT in my lifetime. The order is a bit off using Jikan
        # so checking for N/A score and stopping is not an option.

        url_required_fields = ["id", "mean", "num_scoring_users", "num_list_users", "num_episodes",
                               "average_episode_duration",
                               "media_type", "start_season", "start_date"]

        fields_no_edit_needed = ["id", "mean", "num_scoring_users", "num_list_users"]

        # The fields we need from the JSON object containing information about a single anime.

        anime_data_dict = {'Rows': ['ID', 'Mean Score', 'Scores',
                                    'Members', 'Episodes', 'Duration', 'Type', 'Year', 'Season']}

        page_num = 0
        while not last_show_reached:
            # We loop over pages of anime info that we get from the Jikan API (the info is
            # sorted by score, from highest to lowest) until we reach the lowest rated show.
            # There should be 25 shows per page/batch.

            anime_batch = get_anime_batch_from_MAL(page_num, url_required_fields)
            try:
                print(f"Currently on score {anime_batch['data'][-1]['node']['mean']}")
            except KeyError:
                print("Finished")
            while anime_batch is None:
                # If we failed to get the batch for some reason, we retry until it's a success.
                # Jikan API does not require authorization, so the only reason for a failure
                # could be an outage in the API itself, in which case we wouldn't want to
                # timeout/stop the function as an outage could technically last for a few hours,
                # or even days.
                print("Error - unable to get batch. Sleeping just to be safe, "
                      "then trying again.")
                logging.error("Error - unable to get batch. Sleeping just to be safe, "
                              "then trying again.")
                time.sleep(Sleep.LONG_SLEEP)
                anime_batch = get_anime_batch_from_MAL(page_num, url_required_fields)
                print(anime_batch)

            for anime in anime_batch["data"]:
                if not non_sequels_only or anime_is_valid(anime):
                    title = create_anime_DB_entry(anime, url_required_fields)
                if title.startswith(last_show):
                    last_show_reached = True
                    break
            page_num += 1
        table = pa.Table.from_pydict(anime_data_dict)
        pq.write_table(table, anime_database_name)  # This creates a .parquet file from the dict
        # if self._main_df:
        #     synchronize_main_dbs()
