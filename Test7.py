from general_utils import load_pickled_file
from Test2 import *
from MAL_utils import get_anime_batch_from_jikan
from MAL_utils import MediaTypes
from MAL_utils import Data
import requests
import json
from sortedcollections import OrderedSet


class Tags:

    shows_tags_filename = "shows_tags_dict.pickle"

    _instance = None

    query = '''
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

    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # All properties are loaded on demand
        self._shows_tags_dict = {}
        self._all_tags_list = OrderedSet()

    @property
    def all_tags_list(self):
        if not self._all_tags_list:
            self.get_full_tags_list()
        return self._all_tags_list

    @property
    def shows_tags_dict(self):
        if not self._shows_tags_dict:
            try:
                print("Loading shows-tags dictionary")
                self._shows_tags_dict = load_pickled_file(self.shows_tags_filename)
            except FileNotFoundError:
                print("Shows-tags dictionary not found. Creating new shows-tags dictionary")
                self.get_shows_tags()
        return self._shows_tags_dict

    def get_full_tags_list(self):
        # Gets list of all tags + counts amount of show per studio
        studio_dict = {}

        with open("NSFWTags.txt", 'r') as file:
            banned_tags = file.read().splitlines()

        for show, show_dict in self.shows_tags_dict.items():
            if 'Tags' in show_dict:
                for tag_dict in show_dict['Tags']:
                    if tag_dict['name'] not in banned_tags:
                        self._all_tags_list.add(tag_dict['name'])
            if 'Genres' in show_dict.keys():
                for genre in show_dict['Genres']:
                    self._all_tags_list.add(genre)
            if 'Studio' in show_dict.keys():
                if show_dict['Studio']:
                    if show_dict['Studio'] not in studio_dict:
                        studio_dict[show_dict['Studio']] = 1
                    else:
                        studio_dict[show_dict['Studio']] += 1  # # #

        extra_studios = ['Trigger', 'Passione', 'project No.9', 'POLYGON PICTURES', 'EMT Squared', 'SANZIGEN']

        # Only keeps studios that have over 30 shows or are in extra_studios
        for studio, amount_of_shows in studio_dict.items():
            if amount_of_shows >= 30 or studio in extra_studios:
                self._all_tags_list.add(studio)

        self._all_tags_list = list(self._all_tags_list)

    def get_shows_tags(self):
        def get_recommended_shows():
            try:
                sorted_recs = sorted(media['recommendations']['nodes'], key=lambda x: x['rating'], reverse=True)
            except KeyError:
                return None

            rec_list_length = min(5, len(sorted_recs))
            rec_dict = {}
            for rec in sorted_recs[0:rec_list_length]:
                try:
                    rec_index = ids.index(rec['mediaRecommendation']['idMal'])
                except (TypeError, ValueError):
                    continue
                try:
                    rec_MAL_title = data.titles[rec_index]
                except IndexError:
                    print(rec_index,media)
                # Can't take title directly from the recommendation object,
                # might be different from the MAL title we have in the database
                # rec_dict['Ratings'].append(rec['rating'])
                # rec_dict['Titles'].append(rec_MAL_title)
                rec_dict[rec_MAL_title] = rec['rating']
            return rec_dict

        data=Data()
        url = "https://graphql.anilist.co"
        ids = data.anime_df.row(data.anime_db_stats['ID'])[1:]
        has_next_page = True
        page = 1
        while has_next_page:
            print(f"Currently on page {page}")
            variables = {"page": page, "isAdult": False}
            response = requests.post(url, json={"query": self.query, "variables": variables})
            response = response.json()
            media_list = response["data"]["Page"]["media"]
            # categories = set()
            for media in media_list:
                try:
                    index = ids.index(media["idMal"])
                except (TypeError, ValueError):
                    continue
                title = data.titles[index]  # Check this, might not be synchronized!
                show_stats = data.get_stats_of_shows([title], ["Episodes", "Duration"])
                show_recommendations = get_recommended_shows()

                if show_stats[title]["Episodes"] * show_stats[title]["Duration"] >= 15\
                        and show_stats[title]["Duration"]>=2:
                    result = {
                        "Tags": [{"name": tag["name"], "percentage": tag["rank"], "category": tag["category"]} for tag
                                 in
                                 media["tags"]],
                        "Genres": media["genres"],
                        "Studio": media["studios"]["nodes"][0]["name"] if media["studios"]["nodes"] else None,
                        "Recommended": show_recommendations
                        # If recs amount less than 25 start penalizing?

                    }

                    # Separate studios from tags, genres too maybe?
                    self._shows_tags_dict[title] = result
            has_next_page = response["data"]["Page"]["pageInfo"]["hasNextPage"]
            page = page + 1
            time.sleep(1)

        save_pickled_file(self.shows_tags_filename,self._shows_tags_dict)


# def get_shows_tags():
#
#     def get_recommended_shows():
#         try:
#             sorted_recs = sorted(media['recommendations']['nodes'], key = lambda x:x['rating'], reverse=True)
#         except KeyError:
#             return None
#
#         rec_list_length = min(5, len(sorted_recs))
#         rec_dict={}
#         for rec in sorted_recs[0:rec_list_length]:
#             try:
#                 rec_index = ids.index(rec['mediaRecommendation']['idMal'])
#             except (TypeError, ValueError):
#                 continue
#             rec_MAL_title = data.titles[rec_index] # Can't take title directly from the recommendation object,
#             # might be different from the MAL title we have in the database
#             # rec_dict['Ratings'].append(rec['rating'])
#             # rec_dict['Titles'].append(rec_MAL_title)
#             rec_dict[rec_MAL_title] = rec['rating']
#         return rec_dict
#
#     has_next_page = True
#     page=1
#     while has_next_page:
#         print(f"Currently on page {page}")
#         variables = {"page": page, "isAdult" : False}
#         response = requests.post(url, json={"query": query, "variables": variables})
#         response = response.json()
#         media_list = response["data"]["Page"]["media"]
#         # categories = set()
#         for media in media_list:
#             try:
#                 index = ids.index(media["idMal"])
#             except (TypeError, ValueError):
#                 continue
#             title = data.titles[index] # Check this, might not be synchronized!
#             show_stats = data.get_stats_of_shows([title],["Episodes" , "Duration"])
#             show_recommendations = get_recommended_shows()
#
#             if show_stats[title]["Episodes"]*show_stats[title]["Duration"] >= 15:
#
#                 result = {
#                     "Tags": [{"name": tag["name"], "percentage": tag["rank"], "category": tag["category"]} for tag in
#                              media["tags"]],
#                     "Genres": media["genres"],
#                     "Studio": media["studios"]["nodes"][0]["name"] if media["studios"]["nodes"] else None,
#                     "Recommended" : show_recommendations
#                     # If recs amount less than 25 start penalizing?
#
#                 }
#
#                 # Separate studios from tags, genres too maybe?
#                 show_tags_dict[title] = result
#         has_next_page = response["data"]["Page"]["pageInfo"]["hasNextPage"]
#         page=page+1
#         time.sleep(1)

        # Add stuff like Hanasaku Iroha : HSH separately here?...we can just iterate over titles, if title is
        # TV + long enough, find the ID, request separately on MAL, add to tag_dict. For now leave alone


# def create_partial_anime_db(cols):
#     data.non_sequel_anime_db

# url = "https://graphql.anilist.co"
# show_tags_dict={}
# # data=Data()
# tags=Tags()
# k = tags.shows_tags_dict['Cowboy Bebop']
# data=Data()
# # ids = data.anime_df.rows()[0][1:]
# print(6)
# print(5)
# get_shows_tags()

# create_partial_anime_db(list(show_tags_dict.keys()))


# relevant_titles = list(show_tags_dict.keys())
# relevant_titles = [x for x in relevant_titles if
#                    data.anime_df.filter(pl.col('Rows') == "Duration")[x].item() > 15]
# partial_anime_df = data.anime_df.select(['Rows'] + relevant_titles)
# partial_main_df = data.main_df.select(data.main_db_stats + relevant_titles)
# mean_score_row = partial_anime_df.filter(pl.col('Rows') == "Mean Score")
#
# user_tag_affinity_dict = {}
#
# for row in partial_main_df.iter_rows():
#     pass