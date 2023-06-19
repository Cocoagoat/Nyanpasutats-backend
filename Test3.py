# from general_utils import load_pickled_file
# from Test2 import *
# from MAL_utils import get_anime_batch_from_jikan
# from MAL_utils import MediaTypes
# from MAL_utils import Data
# import requests
# import json
# t = load_pickled_file("test_graph3.pickle")
#
#
#
# # query = '''
# # query ($malId: Int) {
# #   Media(idMal: $malId) {
# #     tags {
# #       name
# #       category
# #       rank
# #     }
# #     genres
# #     studios(isMain: true) {
# #       nodes {
# #         name
# #       }
# #     }
# #   }
# # }
# # '''
#
# query = '''
# query ($page: Int, $isadult : Boolean) {
#   Page(page: $page, perPage: 50) {
#     pageInfo {
#       total
#       perPage
#       currentPage
#       lastPage
#       hasNextPage
#     }
#     media (type : ANIME, isAdult : $isadult) {
#       idMal
#       tags {
#         name
#         category
#         rank
#       }
#       genres
#       studios(isMain : true) {
#         nodes {
#            name
#         }
#       }
#       recommendations(page : 1, perPage : 25){
#         nodes{
#           rating
#           mediaRecommendation{
#             idMal
#             title{
#               romaji
#             }
#           }
#         }
#       }
#     }
#   }
# }
# '''
#
#
# def get_shows_categories():
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
#
#         # Add stuff like Hanasaku Iroha : HSH separately here?...we can just iterate over titles, if title is
#         # TV + long enough, find the ID, request separately on MAL, add to tag_dict. For now leave alone
#
#
# # def create_partial_anime_db(cols):
# #     data.non_sequel_anime_db
#
# url = "https://graphql.anilist.co"
# show_tags_dict={}
# data=Data()
# ids = data.anime_df.rows()[0][1:]
# get_shows_categories()
#
# # create_partial_anime_db(list(show_tags_dict.keys()))
#
# save_pickled_file("shows_categories2.pickle", show_tags_dict)
# # relevant_titles = list(show_tags_dict.keys())
# # relevant_titles = [x for x in relevant_titles if
# #                    data.anime_df.filter(pl.col('Rows') == "Duration")[x].item() > 15]
# # partial_anime_df = data.anime_df.select(['Rows'] + relevant_titles)
# # partial_main_df = data.main_df.select(data.main_db_stats + relevant_titles)
# # mean_score_row = partial_anime_df.filter(pl.col('Rows') == "Mean Score")
# #
# # user_tag_affinity_dict = {}
# #
# # for row in partial_main_df.iter_rows():
# #     pass