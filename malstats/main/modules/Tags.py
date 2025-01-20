import os
import requests
from polars import ColumnNotFoundError
import logging
from .filenames import *
import time
from .AnimeDB import AnimeDB
from .Graphs import Graphs
from sortedcollections import OrderedSet

from .general_utils import load_pickled_file, save_pickled_file, split_list_interval

# Class currently a bit of a mess with out-of-date comments
# due to last-minute daily updating features added, to be refactored

logger = logging.getLogger("nyanpasutats")


class Tags:
    """This class holds various data structures related to Anilist tags. Main data structures :

    - entry_tags_dict - An object containing
     information about each show that meets condition defined in AnimeDB.meets_conditions
      (each show that passes the popularity and score thresholds defined there)

    - show_tags_dict - An object containing information about each show
    considered to be a MAIN show (non-sequel) as decided by the Graphs module.
     See shows_tags_dict definition for more details.

    """
    #  Example pair from entry_tags_dict :
    #
    #   'Cowboy Bebop' : {'Tags': [{'name': 'Space',
    #   'percentage': 0.94,
    #   'category': 'Setting-Universe'},
    #   {'name': 'Crime', 'percentage': 0.92, 'category': 'Theme-Other'},
    #   {'name': 'Episodic', 'percentage': 0.89, 'category': 'Technical'},
    #   {'name': 'Ensemble Cast', 'percentage': 0.86, 'category': 'Cast-Main Cast'},
    #   ...
    #   'Genres': ['Action', 'Adventure', 'Drama', 'Sci-Fi'],
    #   'Studio': 'Sunrise',
    #   'Recommended': {'Samurai Champloo': 1353,
    #   'Trigun': 548,
    #   'Space☆Dandy': 281,
    #   'Black Lagoon': 219,
    #   'Baccano!': 145},
    #   'Main': 'Cowboy Bebop' # Main show, will be the same for every entry related to Cowboy Bebop
    #   'DoubleTags': [{'name': '<Ensemble Cast>x<Space>', 'percentage': 0.86},
    #   {'name': '<Primarily Adult Cast>x<Space>', 'percentage': 0.85},
    #   {'name': '<Tragedy>x<Space>', 'percentage': 0.766},
    #   {'name': '<Male Protagonist>x<Space>', 'percentage': 0.67},
    # ...

    # Example pair from show_tags_dict :
    # 'Cowboy Bebop' :  {'Tags': [{'name': 'Terrorism', 'percentage': 0.97},
    #             {'name': 'Space', 'percentage': 0.94},
    #             {'name': 'Crime', 'percentage': 0.92},
    #             {'name': 'Episodic', 'percentage': 0.89},
    # ...
    #   'DoubleTags': [{'name': '<Action>x<Adventure>', 'percentage': 1},
    #                  {'name': '<Action>x<Drama>', 'percentage': 1},
    #                  {'name': '<Action>x<Sci-Fi>', 'percentage': 1},
    #                  {'name': '<Drama>x<Adventure>', 'percentage': 1},
    #   'Genres': ['Adventure', 'Action', 'Drama', 'Comedy', 'Sci-Fi', 'Mystery'],
    #   'Studios': {'Cowboy Bebop': 'Sunrise',
    #               'Cowboy Bebop: Tengoku no Tobira': 'bones',
    #               'Cowboy Bebop: Yose Atsume Blues': 'Sunrise'},
    #   'Recommended': {'Samurai Champloo': 1353,
    #                   'Trigun': 548,
    #                   'Space☆Dandy': 298,
    #                   'Black Lagoon': 219,
    #                   'Baccano!': 145},
    #   'Related': {'Cowboy Bebop': 1,
    #               'Cowboy Bebop: Tengoku no Tobira': 0.576,
    #               'Cowboy Bebop: Yose Atsume Blues': 0.135}}

    url = "https://graphql.anilist.co"

    page_query = '''
    query ($page: Int, $isadult : Boolean) {
      Page(page: $page, perPage: 50) {
        pageInfo {
          total
          perPage
          currentPage
          lastPage
          hasNextPage
        }
        media (type : ANIME, isAdult : $isadult) {
          title{
            romaji
          }
          idMal
          tags {
            name
            category
            rank
          }
          genres
          studios(isMain : true) {
            nodes {
               name
            }
          }
          recommendations(page : 1, perPage : 25){
            nodes{
              rating
              mediaRecommendation{
                idMal
                title{
                  romaji
                }
              }
            }
          }
        }
      }
    }
    '''

    entry_query = '''
        query ($idMal: Int) {
          Media(type: ANIME, idMal: $idMal) {
            title {
              romaji
            }
            idMal
            tags {
              name
              category
              rank
            }
            genres
            studios(isMain: true) {
              nodes {
                name
              }
            }
            recommendations(page: 1, perPage: 25) {
              nodes {
                rating
                mediaRecommendation {
                  idMal
                  title {
                    romaji
                  }
                }
              }
            }
          }
        }
        '''

    _instance = None

    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._entry_tags_dict = {}
            cls._instance._entry_tags_dict_nls = {}
            cls._instance._entry_tags_dict_updated = {}
            cls._instance._entry_tags_dict_nls_updated = {}
            cls._instance._entry_tags_dict2 = {}
            cls._instance._entry_tags_dict2_updated = {}
            cls._instance._show_tags_dict = {}
            cls._instance._show_tags_dict_nls = {}
            cls._instance._show_tags_dict_updated = {}
            cls._instance._show_tags_dict_nls_updated = {}
            cls._instance._tags_per_category = {}
            cls._instance.anime_db = AnimeDB()
            cls._instance._all_anilist_tags = []
            cls._instance._all_single_tags = []
            cls._instance._all_genres = []
            cls._instance._all_studios = []
            cls._instance._all_doubletags = []
            cls._instance.graphs = Graphs()
            # try:
            #     cls._instance.data = load_pickled_file(data_path / "general_data.pickle")
            # except FileNotFoundError:
            #     cls._instance.data = GeneralData().generate_data()
            cls._instance.tag_types = ["Single", "Double"]
            cls._tags_to_include = []
            cls._banned_tags = cls.get_banned_tags()

            # data = load_pickled_file(data_path / "general_data.pickle")
            # cls.tags_to_include = data.OG_aff_means.keys()
        return cls._instance

    def __init__(self):
        # All properties are loaded on demand
        pass

    @classmethod
    def reset(cls):
        cls._instance = None

    @property
    def tags_to_include(self):
        if not self._tags_to_include and os.path.exists(data_path / "general_data.pickle"):
            data = load_pickled_file(data_path / "general_data.pickle")
            try:
                self._tags_to_include = data.OG_aff_means.keys()
            except AttributeError:
                self._tags_to_include = True
        return self._tags_to_include

    @property
    def entry_tags_dict(self):
        if not self._entry_tags_dict:
            try:
                self._entry_tags_dict = load_pickled_file(entry_tags_filename)
            except FileNotFoundError:
                self.create_entry_tags_dict()
        return self._entry_tags_dict

    @property
    def entry_tags_dict_nls(self):
        if not self._entry_tags_dict_nls:
            try:
                self._entry_tags_dict_nls = load_pickled_file(entry_tags_nls_filename)
            except FileNotFoundError:
                relevant_titles = self.graphs.all_graphs_nls.all_titles
                self._entry_tags_dict_nls = {entry: entry_tags for entry, entry_tags in
                                             self.entry_tags_dict.items() if entry in relevant_titles}
                save_pickled_file(entry_tags_nls_filename, self._entry_tags_dict_nls)
        return self._entry_tags_dict_nls

    @property
    def entry_tags_dict_updated(self):
        if not self._entry_tags_dict_updated:
            try:
                self._entry_tags_dict_updated = load_pickled_file(entry_tags_updated_filename)
            except FileNotFoundError:
                self.update_tags()
        return self._entry_tags_dict_updated

    @property
    def entry_tags_dict_nls_updated(self):
        if not self._entry_tags_dict_nls_updated:
            try:
                self._entry_tags_dict_nls_updated = load_pickled_file(entry_tags_nls_updated_filename)
            except FileNotFoundError:
                self._create_nls_updated()
        return self._entry_tags_dict_nls_updated

    def _create_nls_updated(self):
        relevant_titles = self.graphs.all_graphs_nls_updated.all_titles
        self._entry_tags_dict_nls_updated = {entry: entry_tags for entry, entry_tags in
                                             self.entry_tags_dict_updated.items() if entry in relevant_titles}
        save_pickled_file(entry_tags_nls_updated_filename, self._entry_tags_dict_nls_updated)

    @property
    def entry_tags_dict2(self):
        if not self._entry_tags_dict2:
            self.get_entry_tags2()
        return self._entry_tags_dict2

    @property
    def entry_tags_dict2_updated(self):
        # No need for filename since it gets created
        # very quickly and only needs to load once
        if not self._entry_tags_dict2_updated:
            self.get_entry_tags2(update=True)
        return self._entry_tags_dict2_updated

    @property
    def show_tags_dict(self):
        if not self._show_tags_dict:
            try:
                self._show_tags_dict = load_pickled_file(shows_tags_filename)
            except FileNotFoundError:
                self.get_shows_tags()
        return self._show_tags_dict

    @property
    def show_tags_dict_nls(self):
        if not self._show_tags_dict_nls:
            try:
                self._show_tags_dict_nls = load_pickled_file(shows_tags_nls_filename)
            except FileNotFoundError:
                self.get_shows_tags(no_low_scores=True)
                self._show_tags_dict_nls = load_pickled_file(shows_tags_nls_filename)
                del self._show_tags_dict  # workaround
        return self._show_tags_dict_nls

    @property
    def show_tags_dict_updated(self):
        if not self._show_tags_dict_updated:
            try:
                self._show_tags_dict_updated = load_pickled_file(shows_tags_updated_filename)
            except FileNotFoundError:
                self.get_shows_tags(update=True)
        return self._show_tags_dict_updated

    @property
    def show_tags_dict_nls_updated(self):
        if not self._show_tags_dict_nls_updated:
            try:
                self._show_tags_dict_nls_updated = load_pickled_file(shows_tags_nls_updated_filename)
            except FileNotFoundError:
                # Removing all low-scored shows from the shows dict is a massive pain,
                # easier to just create a new one
                self.get_shows_tags(update=True, no_low_scores=True)

        return self._show_tags_dict_nls_updated

    @property
    def all_anilist_tags(self):
        if not self._all_anilist_tags:
            self._all_anilist_tags = self.all_single_tags + self.all_genres + self.all_studios + self.all_doubletags
            self._all_anilist_tags = [x for x in self._all_anilist_tags if (
                    self.tags_to_include == True or x in self.tags_to_include)]
        return self._all_anilist_tags

    @property
    def all_single_tags(self):
        if not self._all_single_tags:
            self._all_single_tags = self.get_full_tags_list()
        return self._all_single_tags

    @property
    def all_genres(self):
        if not self._all_genres:
            self._all_genres = self.get_full_genres_list()
        return self._all_genres

    @property
    def all_studios(self):
        if not self._all_studios:
            self._all_studios = self.get_full_studios_list()
        return self._all_studios

    @property
    def all_doubletags(self):
        if not self._all_doubletags:
            self._all_doubletags = self.get_full_doubletags_list(sorted=True)
        return self._all_doubletags

    @property
    def tags_per_category(self):
        """

        :return: A dictionary of categories, with each category having a separate group of tags in a list.
        """
        if not self._tags_per_category:
            allowed_categories = ['Cast-Main Cast', 'Cast-Traits', 'Sexual Content', 'Setting-Scene', 'Demographic',
                                  'Setting-Time', 'Setting-Universe']
            self._tags_per_category = {x['category']: [] for entry in self.entry_tags_dict_nls.keys()
                                       for x in self.entry_tags_dict_nls[entry]['Tags']}

            for entry in self.entry_tags_dict_nls.keys():
                for tag in self.entry_tags_dict_nls[entry]['Tags']:
                    if tag['name'] not in self._tags_per_category[tag['category']]:
                        self._tags_per_category[tag['category']].append(tag['name'])

            test = self._tags_per_category
            themes = [x for key, value in self._tags_per_category.items()
                      for x in value if key.startswith("Theme")]
            if "Episodic" in self._tags_per_category['Technical']:
                themes.append("Episodic")
            del self._tags_per_category['Technical']

            leftover_tags = []
            for key in list(self._tags_per_category.keys()):
                if key.startswith("Theme"):
                    del self._tags_per_category[key]

                elif key not in allowed_categories:
                    for tag in self._tags_per_category[key]:
                        leftover_tags.append(tag)
                    del self._tags_per_category[key]

            split_themes = split_list_interval(themes, 10)
            theme_dict = {f"Themes-{i + 1}": split_themes[i] for i in range(len(split_themes))}
            k = 10
            for tag in leftover_tags:
                if k == 0:
                    k = 10
                theme_dict[f"Themes-{k}"].append(tag)
                k -= 1

            doubles = self.get_full_doubletags_list(sorted=True)
            split_doubles = split_list_interval(doubles, 30)
            doubles_dict = {f"Doubles-{i + 1}": split_doubles[i] for i in range(len(split_doubles))}
            doubles_dict2 = {"Doubles": doubles}

            self._tags_per_category = self._tags_per_category | theme_dict | doubles_dict2  # | doubles_per_tag

        return self._tags_per_category

    def get_category_tag_type(self, category):
        """

        :param category: Name of the category
        :return: Either "Single" or "Double"
        """
        try:
            first_tag_name = self.tags_per_category[category][0]
        except KeyError:
            return "Single"  # Genres or Studios will always be single tags
        tag_type = "Single" if first_tag_name and "<" not in first_tag_name else "Double"
        # All tags in a category will be of the same type, only doubletags have "<>" in them.
        return tag_type

    def get_full_tags_list(self):
        tags = OrderedSet()
        for show, show_dict in self.entry_tags_dict_nls.items():
            if 'Tags' in show_dict:
                for tag_dict in show_dict['Tags']:
                    if tag_dict['name'] not in self._banned_tags and (
                            self.tags_to_include == True or tag_dict['name'] in self.tags_to_include):
                        tags.add(tag_dict['name'])
        return list(tags)

    def get_single_tag_counts(self):
        """Goes over each show in the newly created entry_tags_dict to gather every tag that appears at least once,
        # as well as checks how often they appear."""
        tag_counts = {}
        for entry, entry_data in self.entry_tags_dict.items():
            tags_genres = self.gather_tags_and_genres(entry_data)
            for tag in tags_genres:  # Genres are strings, tags are dictionaries with ['name'] and ['percentage']
                tag_name = tag['name'] if type(tag) is dict else tag
                tag_p = tag['percentage'] if type(tag) is dict else 1
                if tag_name not in self._banned_tags:
                    try:
                        tag_counts[tag_name] += tag_p
                    except KeyError:  # New tag, key doesn't exist yet
                        tag_counts[tag_name] = tag_p

        tag_counts = {tag: count for tag, count in
                      sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)}
        return tag_counts

    def get_full_genres_list(self):
        genres = OrderedSet()
        for show, show_dict in self.entry_tags_dict.items():
            if 'Genres' in show_dict.keys():
                for genre in show_dict['Genres']:
                    genres.add(genre)

        genres = [x for x in genres if self.tags_to_include == True or x in self.tags_to_include]
        return list(genres)

    def get_full_studios_list(self):
        studios = OrderedSet()
        studio_dict = {}
        for show, show_dict in self.entry_tags_dict.items():
            if 'Studio' in show_dict.keys():
                if show_dict['Studio']:
                    if show_dict['Studio'] not in studio_dict:
                        studio_dict[show_dict['Studio']] = 1
                    else:
                        studio_dict[show_dict['Studio']] += 1

        extra_studios = ['Trigger', 'Passione', 'project No.9', 'POLYGON PICTURES', 'EMT Squared', 'SANZIGEN']

        for studio, amount_of_shows in studio_dict.items():
            if amount_of_shows >= 30 or studio in extra_studios:
                studios.add(studio)

        studios = [x for x in studios if self.tags_to_include == True or x in self.tags_to_include]
        return list(studios)

    def get_full_doubletags_list(self, sorted=False):
        if not sorted:
            double_tags = OrderedSet()
            for show, show_dict in self.entry_tags_dict.items():
                if 'DoubleTags' in show_dict:
                    for tag_dict in show_dict['DoubleTags']:
                        double_tags.add(tag_dict['name'])
        else:
            double_tag_counts = self.get_double_tag_counts(only_relevant=True)
            double_tags = double_tag_counts.keys()
        double_tags = [x for x in double_tags if self.tags_to_include == True or x in self.tags_to_include]
        return list(double_tags)

    def get_double_tag_counts(self, only_relevant=True):
        # All possible combinations of two tags
        tag_counts = self.get_single_tag_counts()

        # If a tag appears less than 3.45 times (weighted by %) throughout each entry, we disregard it completely.
        relevant_tags = {tag: count for tag, count in tag_counts.items() if count >= 3.45}
        single_tags = list(relevant_tags.keys())

        # If a double tag appears less than 15 times (weighted by %) throughout each entry,
        # we disregard it completely. Since there are ~50k double tags we only want the most important ~2k tags,
        # otherwise it'll be cluttered with tags that almost never appear together + take ages to run.
        double_tag_counts = {f"<{tag1}>x<{single_tags[j]}>": 0 for i, tag1 in
                             enumerate(single_tags) for
                             j in range(i + 1, len(single_tags))}
        # Calculate how many times each tag pair appears in each entry (weighted).
        for entry, entry_data in self._entry_tags_dict.items():
            tags_genres = self.gather_tags_and_genres(entry_data)
            for i, tag1 in enumerate(tags_genres):
                tag1_name = tag1['name'] if type(tag1) is dict else tag1
                if tag1_name not in single_tags:
                    continue
                tag1_p = tag1['percentage'] if type(tag1) is dict else 1
                for j in range(i + 1, len(tags_genres)):
                    tag2 = tags_genres[j]
                    tag2_name = tag2['name'] if type(tag2) is dict else tag2
                    if tag2_name not in single_tags:
                        continue
                    tag2_p = tag2['percentage'] if type(tag2) is dict else 1
                    try:
                        double_tag_counts[f"<{tag1_name}>x<{tag2_name}>"] += min(tag1_p, tag2_p)
                    except KeyError:
                        double_tag_counts[f"<{tag2_name}>x<{tag1_name}>"] += min(tag1_p, tag2_p)
        if only_relevant:
            relevant_double_tags = list({tag: count for tag, count
                                         in double_tag_counts.items() if count >= 15}.keys())
            double_tag_counts = {tag: count for tag, count in sorted(
                double_tag_counts.items(), key=lambda x: x[1], reverse=True) if tag in relevant_double_tags}

        return double_tag_counts

    def _get_shows_from_page(self, page_num):
        def fetch_response():
            try:
                response = requests.post(self.url,
                                         json={"query": self.page_query,
                                               "variables": variables},
                                         timeout=30).json()
                time.sleep(3)
                return response
            except requests.exceptions.ReadTimeout:
                print(f"Request for fetching tags page {page_num}"
                      f" timed out. Retrying.")
                logger.warning(f"Request for fetching tags page {page_num}"
                               f" timed out. Retrying.")

                return

        variables = {"page": page_num, "isAdult": False}

        ok_response_received = False
        while not ok_response_received:
            response = False
            while not response:
                response = fetch_response()

            try:
                show_list = response["data"]["Page"]["media"]
                has_next_page = response["data"]["Page"]["pageInfo"]["hasNextPage"]
                ok_response_received = True
            except:
                try:
                    error_code = response['error'][0]['status']
                except (IndexError, KeyError):
                    error_code = response['error']['status']
                if error_code != 999:
                    print(f"Error getting shows from page {page_num} while fetching tags"
                          f".Error code is {error_code}. Response is {response}. Sleeping 300 seconds.")
                    logger.error(f"Error getting shows from page {page_num} while fetching tags"
                                 f".Error code is {error_code}. Response is {response}. Sleeping 300 seconds.")
                    time.sleep(300)
                else:
                    show_list = []
                    has_next_page = False
                    ok_response_received = True  # last page
        return show_list, has_next_page

    def _get_recommended_shows(self, entry):
        try:
            sorted_recs = sorted(entry['recommendations']['nodes'], key=lambda x: x['rating'], reverse=True)
        except KeyError:
            return None

        rec_list_length = min(5, len(sorted_recs))
        rec_dict = {}
        for rec in sorted_recs[0:rec_list_length]:
            try:
                rec_MAL_title = self.anime_db.id_title_map[rec['mediaRecommendation']['idMal']]
            except KeyError:
                continue

            # try:
            #     rec_index = self.anime_db.ids.index(rec['mediaRecommendation']['idMal'])
            # except (TypeError, ValueError):
            #     continue
            # try:
            #     rec_MAL_title = self.anime_db.titles[rec_index]
            # except IndexError:
            #     continue
            # Can't take title directly from the recommendation object,
            # might be different from the MAL title we have in the database
            rec_dict[rec_MAL_title] = rec['rating']
        return rec_dict

    def _get_entry_data(self, title, update=False):

        malId = self.anime_db.get_id_by_title(title)
        variables = {"idMal": malId}
        logger.info(f"Fetching entry {title} with malId {malId}")
        entry = requests.post(self.url,
                              json={"query": self.entry_query,
                                    "variables": variables},
                              timeout=30).json()
        logger.info(f"Successfully fetched entry data for {title}")
        time.sleep(2.5)
        entry = entry['data']['Media']
        

        # title = self._get_entry_title(entry, ids)
        try:
            show_recommendations = self._get_recommended_shows(entry)
            logger.info(f"Successfully fetched recommendations for {title}")
        except Exception:
            logger.error(f"Error getting recommendations for entry {title}")
            show_recommendations = {}

        
        graphs_dict = self.graphs.all_graphs if not update else self.graphs.all_graphs_updated
        try:
            main_entry, _ = graphs_dict.find_related_entries(title)
            logger.info(f"Successfully found main entry {title}")
        except Exception:
            print(f"Error finding main entry in graphs for entry {title}.")
            logger.error(f"Error finding main entry in graphs for entry {title}.")
            main_entry = title

        entry_data = {
            "Tags": [{"name": tag["name"], "percentage": self.adjust_tag_percentage(tag["rank"]),
                      "category": tag["category"]} for tag in entry["tags"]
                     if tag["rank"] >= 60 and tag['name'] not in self._banned_tags],
            "Genres": entry["genres"],
            "Studio": entry["studios"]["nodes"][0]["name"] if entry["studios"]["nodes"] else None,
            "Recommended": show_recommendations,
            "Main": main_entry
            # If recs amount less than 25 start penalizing?
        }
        logger.info(f"Successfully created entry_data for {title}. Returning")
        return entry_data

    # def _get_entry_title(self, entry):
    #     try:
    #         return self.anime_db.id_title_map[entry["idMal"]]
    #     except KeyError:
    #         return
    #     # title = self.anime_db.titles[index]
    #     # return title

    def _add_single_tags_to_entry_tags_dict(self, update=False):
        # ids = self.anime_db.df.row(self.anime_db.stats['ID'])[1:]
        has_next_page = True
        page_num = 1 if not update else 350
        entry_tags_dict = self._entry_tags_dict if not update else self._entry_tags_dict_updated
        relevant_shows = self.graphs.all_graphs.all_titles if not update else self.graphs.all_graphs_updated.all_titles

        while has_next_page:
            print(f"Currently on page {page_num}")
            entry_list, has_next_page = self._get_shows_from_page(page_num)
            for entry in entry_list:
                try:
                    # title = self._get_entry_title(entry)
                    title = self.anime_db.id_title_map[entry["idMal"]]
                    if title in relevant_shows:
                        entry_data = self._get_entry_data(entry)
                except (TypeError, ValueError, KeyError):
                    continue
                entry_tags_dict[title] = entry_data
            page_num += 1
            time.sleep(1)

    def _add_double_tags_to_entry_tags_dict(self, update=False, relevant_titles=None):
        entry_tags_dict = self._entry_tags_dict if not update else self._entry_tags_dict_updated

        relevant_double_tag_counts = self.get_double_tag_counts(only_relevant=True)
        relevant_double_tags = list(relevant_double_tag_counts.keys())

        for entry, entry_data in entry_tags_dict.items():
            if relevant_titles is not None and entry not in relevant_titles:
                continue
            tags_genres = self.gather_tags_and_genres(entry_data)
            entry_data['DoubleTags'] = []
            for i, tag1 in enumerate(tags_genres):
                tag1_name = tag1['name'] if type(tag1) is dict else tag1
                tag1_p = tag1['percentage'] if type(tag1) is dict else 1

                for j in range(i + 1, len(tags_genres)):
                    tag2 = tags_genres[j]
                    tag2_name = tag2['name'] if type(tag2) is dict else tag2
                    tag2_p = tag2['percentage'] if type(tag2) is dict else 1

                    if f"<{tag1_name}>x<{tag2_name}>" in relevant_double_tags:
                        entry_tags_dict[entry]['DoubleTags'].append(
                            {'name': f"<{tag1_name}>x<{tag2_name}>", 'percentage': min(tag1_p, tag2_p)})

                    elif f"<{tag2_name}>x<{tag1_name}>" in relevant_double_tags:
                        entry_tags_dict[entry]['DoubleTags'].append(
                            {'name': f"<{tag2_name}>x<{tag1_name}>", 'percentage': min(tag1_p, tag2_p)})

            entry_data['DoubleTags'] = sorted(entry_data['DoubleTags'], reverse=True, key=lambda x: x['percentage'])

    def create_entry_tags_dict(self):

        self._add_single_tags_to_entry_tags_dict()
        # Problematic case due to how it's listed in the MAL API
        if 'Black Clover: Mahou Tei no Ken' in self._entry_tags_dict.keys():
            del self._entry_tags_dict['Black Clover: Mahou Tei no Ken']

        # Add the double tags to the data dict of each entry
        self._add_double_tags_to_entry_tags_dict()
        save_pickled_file(entry_tags_filename, self._entry_tags_dict)

    def get_entry_tags2(self, update=False):
        entry_tags_dict2 = {}
        entry_tags_dict = self.entry_tags_dict if not update else self.entry_tags_dict_updated

        for entry, entry_data in entry_tags_dict.items():
            entry_tags_dict2[entry] = {}
            entry_tags_list = entry_data['Tags'] + entry_data['DoubleTags']
            for tag in entry_tags_list:
                if "<" in tag['name']:
                    entry_tags_dict2[entry][tag['name']] = {'percentage':
                                                                tag['percentage']}
                else:
                    entry_tags_dict2[entry][tag['name']] = {'percentage':
                                                                tag['percentage'], 'category': tag['category']}
            genres_studios = entry_data['Genres']
            if entry_data['Studio'] in self.all_anilist_tags:
                genres_studios = genres_studios + [entry_data['Studio']]

            for tag in genres_studios:
                entry_tags_dict2[entry][tag] = {'percentage': 1}
        if update:
            self._entry_tags_dict2_updated = entry_tags_dict2
        else:
            self._entry_tags_dict2 = entry_tags_dict2

    def get_shows_tags(self, no_low_scores=False, update=False):
        def calc_length_coeff(entry, stats):
            return round(min(1, stats[entry]["Episodes"] *
                             stats[entry]["Duration"] / 200), 3)

        def get_tags_from_entries(double=False):
            show_tags = {}
            if not double:
                all_entries_tags_list = [{'name': tag['name'], 'percentage': tag['percentage']}
                                         for entry in show_related_entries
                                         for tag in entry_tags_dict[entry]['Tags']]
            else:
                all_entries_tags_list = [{'name': double_tag['name'], 'percentage': double_tag['percentage']}
                                         for entry in show_related_entries
                                         for double_tag in entry_tags_dict[entry]['DoubleTags']]
            for tag in all_entries_tags_list:
                if tag['name'] not in show_tags or tag['percentage'] > show_tags[tag['name']]:
                    # Each tag a show has gets the maximum value of said tag among its entries
                    show_tags[tag['name']] = tag['percentage']
            show_tags = {tag[0]: tag[1] for tag in sorted(show_tags.items(),
                                                          key=lambda x: x[1], reverse=True)}
            return show_tags

        def get_recommended_shows():
            show_recommended = {}
            for entry in show_related_entries:
                entry_recommended = entry_tags_dict[entry]['Recommended']
                for rec_entry, rec_value in entry_recommended.items():
                    if rec_entry not in show_recommended:
                        show_recommended[rec_entry] = rec_value
                    else:
                        show_recommended[rec_entry] += rec_value

            sorted_recommended_shows = sorted(show_recommended.items(),
                                              reverse=True, key=lambda x: x[1])[0:5]
            show_recommended = {rec[0]: rec[1] for rec in sorted(sorted_recommended_shows,
                                                                 key=lambda x: x[1], reverse=True)}
            return show_recommended

        if update:
            entry_tags_dict = self.entry_tags_dict_nls_updated if no_low_scores else self.entry_tags_dict_updated
        else:
            entry_tags_dict = self.entry_tags_dict_nls if no_low_scores else self.entry_tags_dict

        show_tags_dict = {}
        graphs = self.graphs.all_graphs_updated if update else self.graphs.all_graphs
        processed_titles = []
        show_amount = len(entry_tags_dict.keys())
        anime_db = AnimeDB(anime_database_updated_name if update else None)
        for i, entry in enumerate(entry_tags_dict.keys()):

            if i % 500 == 0:
                print(f"Currently on show {i} of {show_amount}")

            if entry in processed_titles:
                continue

            try:
                main_entry, show_related_entries = graphs.find_related_entries(entry)
            except TypeError:
                continue  # Some entries exist on MAL, but not on Anilist (and thus they do not appear
            # in the tags dictionary which is built using Anilist's tags). Those entries
            # are all either recaps, commercials or other side entries that we normally
            # would not want to analyze either way.

            show_related_entries = [entry for entry in show_related_entries
                                    if entry in entry_tags_dict.keys()]

            show_tags_dict[main_entry] = {}

            # Tags (get the tags of all entries, if several entries have the same tag at different
            # percentages then take the highest percentage among them as the show's tag percentage)
            show_tags = get_tags_from_entries(double=False)
            show_tags_dict[main_entry]['Tags'] = [{'name': tag_name, 'percentage': tag_p}
                                                        for tag_name, tag_p in show_tags.items()]

            show_double_tags = get_tags_from_entries(double=True)
            show_tags_dict[main_entry]['DoubleTags'] = [{'name': tag_name, 'percentage': tag_p}
                                                              for tag_name, tag_p in show_double_tags.items()]

            # Genres (simply add all the unique genres of all the entries)
            show_genres = list(set([genre for entry in show_related_entries
                                    for genre in entry_tags_dict[entry]['Genres']]))
            show_tags_dict[main_entry]['Genres'] = show_genres

            # Studios (same as genres)
            show_studios = {entry: entry_tags_dict[entry]['Studio'] for entry in show_related_entries}
            show_tags_dict[main_entry]['Studios'] = show_studios

            # Recommended (we take the recommendations for each entry the show has, sum up the values,
            # and take the top 5 with the highest recommendation values).
            show_recommended = get_recommended_shows()

            show_tags_dict[main_entry]['Recommended'] = show_recommended

            # Related
            stats_of_related_entries = anime_db.get_stats_of_shows(show_related_entries,
                                                                   ["Episodes", "Duration"])
            related = {}
            for entry in show_related_entries:
                length_coeff = calc_length_coeff(entry, stats_of_related_entries)
                related[entry] = length_coeff
            show_tags_dict[main_entry]['Related'] = related

            # # Add main show to entry_dict
            for entry in show_related_entries:
                entry_tags_dict[entry]['Main'] = main_entry

            # To avoid repeat processing
            processed_titles = processed_titles + show_related_entries

        if update:
            if no_low_scores:
                filename = shows_tags_nls_updated_filename
                self._show_tags_dict_nls_updated = show_tags_dict
            else:
                filename = shows_tags_updated_filename
                self._show_tags_dict_updated = show_tags_dict
        else:
            if no_low_scores:
                filename = shows_tags_nls_filename
                self._show_tags_dict_nls = show_tags_dict
            else:
                filename = shows_tags_filename
                self._show_tags_dict = show_tags_dict

        save_pickled_file(filename, show_tags_dict)

    @staticmethod
    def get_tag_index(tag_name, show_tags_list):
        for i, tag in enumerate(show_tags_list):
            if tag_name == tag['name']:
                return i
        return -1

    # def get_avg_score_of_show(self,show):
    # Currently not in use
    #     entry_count = 1
    #     avg = self.data.mean_score_per_show[show]#.item()
    #     for entry, length_coeff in self.show_tags_dict[show]['Related'].items():
    #         if length_coeff == 1 and entry != show:
    #             avg += self.data.mean_score_per_show[entry]#.item()
    #             entry_count += 1
    #     return round(avg/entry_count,2)

    def get_max_score_of_show(self, show, scores):
        """

        :param show: Name of the show.
        :param scores: A dictionary of all the user's scores. If left empty, it means we want
        to calculate the max MAL score of the show (among all its entries, for example maximum
        score of My Hero Academia would be the 6th season's score as of 2023). If not, then the max user
        score of all the show's entries.
        :return:
        """

        max_score = scores[show]  # .item()
        for entry, length_coeff in self.show_tags_dict_nls_updated[show]['Related'].items():
            if length_coeff == 1 and entry != show:
                try:
                    max_score = max(scores[entry], max_score)
                except (ColumnNotFoundError, TypeError) as e:
                    continue
        return max_score

    @staticmethod
    def adjust_tag_percentage(p):
        # p = tag['percentage']
        if p >= 85:
            adjusted_p = round((p / 100), 3)
        elif p >= 70:
            second_coeff = 20 / 3 - p / 15  # Can be a number from 1 to 1.5
            adjusted_p = round((p / 100) ** second_coeff, 3)
            # 85 ---> 1, 70 ---> 2, 15 forward means -1 so m=-1/15, y-2 = -1/15(x-70) --> y=-1/15x + 20/3
        elif p >= 60:
            second_coeff = 16 - p / 5
            adjusted_p = round((p / 100) ** second_coeff, 3)
        else:
            adjusted_p = 0
        return adjusted_p

    def gather_tags_and_genres(self, show_dict):
        """ Gather tags and genres from an entry """
        tags_genres = []
        # banned_tags = self.get_banned_tags()
        if 'Tags' in show_dict:
            tags_genres.extend([tag for tag in show_dict['Tags'] if tag['name'] not in self._banned_tags])
        if 'Genres' in show_dict:
            tags_genres.extend(show_dict['Genres'])
        return tags_genres

    @staticmethod
    def get_banned_tags():
        with open(data_path / "NSFWTags.txt", 'r') as file:
            banned_tags = file.read().splitlines()
        return banned_tags

    @staticmethod
    def format_doubletag(input_string):
        """Removes the angle brackets from doubletags"""
        formatted_string = input_string.replace('<', ' ').replace('>', ' ').strip()
        return formatted_string

    def update_tags(self, update_from_scratch=False):

        titles_to_add, titles_to_remove = AnimeDB.get_post_update_changed_titles(
            update_from_scratch=update_from_scratch)

        self.anime_db = AnimeDB(anime_database_updated_name, reset=True)
        # ids = self.anime_db.df.row(self.anime_db.stats['ID'])[1:]
        # print(f"Length of updated df inside Tags : {len(self.anime_db.df.columns)}")
        # print(f"Titles before filtering inside Tags: {titles_to_add}")
        titles_to_add = self.anime_db.filter_titles(AnimeDB.show_meets_standard_conditions,
                                                    titles_to_add)
        # print(f"Titles after filtering inside Tags: {titles_to_add}")
        print("Beginning graphs update")
        self.graphs.update_graphs(titles_to_add, titles_to_remove, update_from_scratch)
        Graphs.reset()
        print("Finished graphs update")

        if os.path.exists(entry_tags_updated_filename) and not update_from_scratch:
            entry_tags_dict = self.entry_tags_dict_updated
        else:
            entry_tags_dict = self.entry_tags_dict

        print("Beginning entry-tags update")
        print("Adding single tags")

        errors = 0
        for title in titles_to_add:
            try:
                entry_data = self._get_entry_data(title, update=True)
            except (TypeError, ValueError):
                logger.error(f"Error fetching entry {title}.")
                errors += 1
                continue
            except TimeoutError:
                logger.error(f"Operation timed out while fetching entry {title}. Retrying.")
                entry_data = None
                for i in range(10):
                    try:
                        entry_data = self._get_entry_data(title, update=True)
                        break
                    except TimeoutError:
                        logger.error(f"Operation timed out while fetching entry {title}."
                                     f" Sleeping then retrying.")
                        time.sleep(2 ** (i + 4))
                        continue
                if not entry_data:
                    logger.critical(f"Timed out while trying to fetch entry"
                                    f" {title} 10 times in a row. Manual intervention required.")
                    raise TimeoutError(f"Timed out while trying to fetch entry"
                                       f" {title} 10 times in a row. Manual intervention required.")
                    
            if errors == 5:
                raise ValueError("Failed to fetch 5 entries during update.")
            entry_tags_dict[title] = entry_data

        self._entry_tags_dict_updated = entry_tags_dict
        print("Adding double tags")
        self._add_double_tags_to_entry_tags_dict(update=True, relevant_titles=titles_to_add)

        print("Removing old titles from entry tags")
        for title in titles_to_remove:
            self._entry_tags_dict_updated.pop(title, 0)

        print("Saving entry tags")
        save_pickled_file(entry_tags_updated_filename, self._entry_tags_dict_updated)

        self._create_nls_updated()
        print("Beginning show tags update")
        self.get_shows_tags(update=True)
        self.get_shows_tags(update=True, no_low_scores=True)
        print("Finished show tags update")
