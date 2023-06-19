import json
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import polars as pl
from general_utils import *
import dask.dataframe as dd
from MAL_utils import *
import requests
# from datetime import datetime
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from multiprocessing import freeze_support
from abc import ABC, abstractmethod
@timeit
def fetch(url, headers=None):
    if headers is None:
        headers = {}
    response = requests.get(url, headers=headers)
    return response


# global headers
# response = []


@timeit
def fetch_concurrently(urls, num_threads, headers=None):
    responses = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit the fetch function with corresponding url and headers to the ThreadPoolExecutor
        futures = [executor.submit(fetch, url, headers) for url in urls]

        # Collect the results as they become available
        for future in as_completed(futures):
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                print(f"Request failed: {e}")

    return responses


def fetch_concurrently2(urls, num_threads, headers=None):
    responses = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit the fetch function with corresponding url and headers to the ThreadPoolExecutor
        futures = [executor.submit(fetch, url, headers) for url in urls]

        # Collect the results as they become available
        for future in as_completed(futures):
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                print(f"Request failed: {e}")

    return responses


@timeit
def fetch_sequentially(urls,headers):
    responses=[]
    for url in urls:
        response = fetch(url, headers=headers)
        responses.append(response)
    return responses


@timeit
def test_pickle_speed():
    with open('test_dict2.pickle', 'rb') as f:
        loaded_data2 = pickle.load(f)
        return loaded_data2


@timeit
def test_dict_speed(test_dict):
    t1 = perf_counter()
    for i in range(len(test_dict)):
        comparison_list =  test_dict[f'column_{i}']
        time_at_current_point(t1, f"Checkpoint {i}")


# url = 'https://graphql.anilist.co'
#
# query = '''
# query ($idMal: Int) {
#   Media(idMal: $idMal) {
#     id
#     title {
#       romaji
#       english
#       native
#     }
#     format
#     duration
#     episodes
#     relations {
#       edges {
#         node {
#           id
#           title {
#             romaji
#             english
#             native
#           }
#           format
#           duration
#         }
#       }
#     }
#   }
# }
# '''
#
# variables = {
#     'idMal': 1
# }
#
# response = requests.post(url, json={'query': query, 'variables': variables})
# response = response.json()['data']['Media']
# print(response)
# print(response["relations"]["edges"])
# for x in response["relations"]["edges"]:
#     print(x)
#
# terminate_program()

class Title(ABC):
    _instances = {}

    def __new__(cls, title,*args,**kwargs):
        if title in cls._instances:
            return cls._instances[title]
        instance = super().__new__(cls)
        cls._instances[title] = instance
        return instance

    def __str__(self):
        return self.title

    def __init__(self,title,media_type=None):
        if not hasattr(self, '_initialized'):
            self.title=title
            self.media_type = media_type
            self._initialized=True


class RootAnime(Title):
    def __init__(self,title,media_type=None):
        if not hasattr(self, '_initialized'):
            super().__init__(title,media_type)
            self.related_list = []
            self._initialized=True

    def add_related(self,title):
        self.related_list.append(title)


class NonRootAnime(Title):

    def __init__(self,title,media_type=None,relation_type=None,root=None):
        if not hasattr(self, '_initialized'):
            super().__init__(title,media_type)
            self.root = root
            self.relation_type = relation_type
            self._initialized=True


def is_root_title(anime, current_titles):
    # for related_anime in anime["relations"]["edges"]:
    #     if related_anime["node"]["title"]["romaji"] in non_root_titles:
    #         print(Fore.RED + f'Related title previously added - current title is not a root')
    #         return False, related_anime
    #         # This means that a related anime is already in our database, and the one
    #         # we are currently checking can't be the root title
    # print(Fore.GREEN + f'Related titles not found, current title is the root of its tree')
    # return True, None

    for related_anime in anime["related_anime"]:
        if related_anime["node"]["title"] in current_titles:
            print(Fore.RED + f'Related title previously added - current title is not a root')
            return False, related_anime
            # This means that a related anime is already in our database, and the one
            # we are currently checking can't be the root title
    print(Fore.GREEN + f'Related titles not found, current title is the root of its tree')
    return True, None


def initial_root_check(anime, related_anime):
    allowed_types=['parent_story','alternative_setting', 'alternative_version', 'other']
    allowed_media_types=['tv']
    if related_anime["relation_type"] in allowed_types and anime["media_type"] in allowed_media_types\
                        and anime["average_episode_duration"] > 15:
        print(Fore.GREEN + f"{anime['title']} is a {related_anime['relation_type']} of"
              f"{related_anime['node']['title']} and is a new root")
        return True
    print(Fore.RED + f"{anime['title']} is a {related_anime['relation_type']} of"
                       f"{related_anime['node']['title']} and is not a new root")
    return False


def find_closest_relationship(related_anime_list):
    for anime in related_anime_list:
        if anime['relation_type'] == 'prequel' or anime['relation_type'] == 'sequel':
            return anime
    return related_anime_list[0]


def root_child_or_grandchild(anime, root_titles, non_root_titles):
    related_anime_list=[]

    # Implement initial check here - OVAs cannot be roots, neither can "other" or whatever.
    # Only allowed types...but what about Tamayura?

    for related_anime in anime["related_anime"]:
        if related_anime["node"]["title"] in root_titles and not initial_root_check(anime,related_anime):
            related_anime_list.append(related_anime)
            print(Fore.GREEN + f'{anime["title"]} is a direct descendant of {related_anime["node"]["title"]}')
        if len(related_anime_list) == 1:
            return 'child', related_anime
        elif len(related_anime_list) > 1:
            return 'child', find_closest_relationship(related_anime_list)
        if related_anime["node"]["title"] in non_root_titles:
            # This means that a related anime is already in our database, and the one
            # we are currently checking can't be the root title
            print(Fore.RED + f'{anime["title"]} is a direct descendant of {related_anime["node"]["title"]}')
            return 'grandchild',related_anime

    print(f'Ancestor not found, {anime["title"]} is a root title')
    return 'root', None

def is_short(anime):
    # try:
    #     anime_duration = anime["episodes"] * anime["duration"]
    # except TypeError:
    #     print(Fore.RED + f'INVALID - unknown duration')
    #     return True # If duration or amount of episodes are undefined it's definitely an old short, or a completely
    #                 # unknown anime.
    # if (anime_duration < 30):
    #     print(Fore.RED + f'INVALID - {anime["title"]["romaji"]} duration is only {anime_duration} minutes')
    #     return True
    # print(Fore.GREEN + f'VALID - {anime["title"]["romaji"]} duration is {anime_duration} minutes')
    # return False

    try:
        anime_duration = (anime["num_episodes"] * anime["average_episode_duration"])/60
    except TypeError:
        print(Fore.RED + f'INVALID - unknown duration')
        return True # If duration or amount of episodes are undefined it's definitely an old short, or a completely
                    # unknown anime.
    if (anime_duration < 30):
        print(Fore.RED + f'INVALID - {anime["title"]} duration is only {anime_duration} minutes')
        return True
    print(Fore.GREEN + f'VALID - {anime["title"]} duration is {anime_duration} minutes')
    return False



def find_MAL_title_by_ID(ID, ids_titles_years):
    for id,title,years in ids_titles_years:
        if id == ID:
            return title


def create_root_title():
    MAL_title_of_root = find_MAL_title_by_ID(current_anime["id"], ids_titles_years)
    new_root = RootAnime(MAL_title_of_root, current_anime["format"])
    roots.append(new_root)


def create_non_root_title(anime_json, related_anime_json, direct_sibling_of_root):
    title = anime_json["title"]["romaji"]
    media_type = anime_json["format"]
    relation_type = anime_json["relationType"]
    relative_title = related_anime_json["title"]["romaji"]
    new_anime = NonRootAnime(title,media_type,relation_type)
    if direct_sibling_of_root:
        relative = RootAnime(relative_title)
        new_anime.root = relative #Change to relative's title?
    else:
        relative = NonRootAnime(relative_title)
        new_anime.root = relative.root
    return new_anime




# k=RootTitle('Pain','TV')
# k.add_related('Pain S2')
# t=RootTitle('Pain')
# terminate_program()

if __name__ == '__main__':
    freeze_support()
    url = 'https://graphql.anilist.co'
    data=Data()
    pl.Config.set_tbl_cols(50)
    pl.Config.set_tbl_rows(30)

    query = '''
            query ($idMal: Int) {
              Media(idMal: $idMal) {
                id
                title {
                  romaji
                  english
                  native
                }
                format
                duration
                episodes
                relations {
                  edges {
                    node {
                      id
                      title {
                        romaji
                        english
                        native
                      }
                      format
                      duration
                    }
                    relationType
                  }
                }
              }
            }
            '''

    ids = data.anime_df.row(0)[1:]
    release_years = data.anime_df.row(5)[1:]

    ids_titles_years = [x for x in list(zip(ids, data.titles, release_years)) if x[2] is not None and x[2]>1960]
    ids_titles_years_sorted = sorted(ids_titles_years, key=lambda x :x[2])

    roots= []
    non_roots=[]
    root_titles=[]
    non_root_titles=[]

    # agg_dict={}

    for ID, MAL_title, year in ids_titles_years_sorted:
        # url = f'https://api.myanimelist.net/v2/anime/{int(ID)}?fields=id,media_type,mean,' \
        #                    f'related_anime,genres,' \
        #            f'average_episode_duration,num_episodes,num_scoring_users'
        # if MAL_title in non_root_titles:
        #     continue
        variables = {
            'idMal': int(ID)
        }
        print(Fore.LIGHTWHITE_EX + f"Currently on title {MAL_title} from year {year}")
        response = requests.post(url, json={'query': query, 'variables': variables})
        time.sleep(1)
        # anime = call_function_through_process(get_search_results, url)
        while response is None: #turn this into decorator later
            logger.warning("Anime was returned as None, sleeping and retrying")
            time.sleep(Sleep.MEDIUM_SLEEP)  # Just in case
            print("Anime was returned as None, sleeping and retrying")
            response = requests.post(url, json={'query': query, 'variables': variables})
        # print(response.json())

        current_anime = response.json()['data']['Media']
        short = is_short(current_anime)
        if short:
            continue#and anime["title"]["romaji"] not in:

        is_root, related_found = is_root_title(current_anime, non_root_titles)
        # non_root_titles_anilist.append(anime["title"]["romaji"])

        if is_root:

            MAL_title_of_root = find_MAL_title_by_ID(current_anime["id"],ids_titles_years)
            current_anime_obj = RootAnime(MAL_title_of_root, current_anime["format"])
            root_titles.append(current_anime["title"]["romaji"])
            roots.append(current_anime_obj)

            for related_anime in current_anime["relations"]["edges"]:
                print(Fore.LIGHTWHITE_EX + f"Related show : {related_anime}")
                media_type = related_anime["node"]["format"]
                relation_type = related_anime['relationType']

                if media_type!= 'MANGA':

                    MAL_title = find_MAL_title_by_ID(related_anime["node"]["id"], ids_titles_years)
                    new_anime_obj = create_non_root_title(related_anime["node"], current_anime, direct_sibling_of_root=True)
                    RootAnime(MAL_title_of_root).related_list.append(MAL_title)

                    non_root_titles.append(new_anime_obj.title) #ADD ONLY NON_ROOTS!
                    non_roots.append(new_anime_obj)
                    # agg_dict[title].append(MAL_title)
                    # Can't take the title from Anilist directly because it might be different from the MAL title
                    # agg_dict[f"{title} media_type"].append(media_type)
                    # agg_dict[f"{title} relation_type"].append(relation_type)

        else:
            # If the anime is not a root, then it's not a direct sibling of the root either - because all direct
            # siblings of previous roots were already added.
            new_anime_obj = NonRootAnime()
                # for related_anime in anime["relations"]["edges"]:
                #     print(related_anime)
                #     media_type = related_anime["node"]["format"]
                #     relation_type = related_anime['relationType']
                #     if media_type!= 'MANGA':
                #         MAL_title = find_MAL_title_by_ID(related_anime["node"]["id"], ids_titles_years)
                #         # agg_dict[title].append(MAL_title)
                #         # # Can't take the title from Anilist directly because it might be different from the MAL title
                #         # agg_dict[f"{title} media_type"].append(media_type)
                #         # agg_dict[f"{title} relation_type"].append(relation_type)
                #     non_root_titles.append(related_anime["node"]["title"])


# for x in response["relations"]:
#     print(x)
# response = json.dumps(response.json(), indent=2)


#     root = None
#     root_related=[]
#     if title not in non_root_titles:
#         mal_id = data.anime_df[title][1]
#         url = f'https://api.myanimelist.net/v2/anime/{mal_id}?fields=id,media_type,mean,' \
#               f'related_anime,genres,' \
#               f'average_episode_duration,num_episodes,num_scoring_users'
#         anime = call_function_through_process(get_search_results, url)
#         root = find_root_title(title)
#         root_titles.append(root) b

# data.main_df

# test_dict = test_pickle_speed()
# test_dict_speed(test_dict)
# data=Data()
# print(data.anime_df)
# print(int(data.anime_df['Steins;Gate'][0]))
# transpose_df = data.main_df.transpose()
# print(5)
#
# print(5)
# for col in data.anime_df.columns[1:4]:
#     variables = {'idMal': int(data.anime_df[col][0])}
#     query = """
#     query ($search: String, $idMal: Int) {
#       Media (search: $search, idMal: $idMal, type: ANIME) {
#         id
#         title {
#           romaji
#           english
#           native
#         }
#         coverImage {
#           large
#         }
#         tags{
#           name
#           rank
#           category
#           id
#         }
#         genres
#         relations{
#           edges{
#             relationType
#             characterName
#             staffRole
#             favouriteOrder
#           }
#           nodes{
#             title{
#               romaji
#             }
#           }
#           pageInfo{
#             total
#             perPage
#             currentPage
#             lastPage
#             hasNextPage
#           }
#
#         }
#       }
#     }
#     """
#     payload = {"query": query, "variables": variables}
#     response = requests.post(url, headers=headers, data=json.dumps(payload))
#     print(json.dumps(response.json(), indent=2))


# url = 'https://api.jikan.moe/v4/users/BaronBrixius'
# data = get_search_results(url)
# print(data)
# print(data["data"]["joined"])
# cols2 = data.anime_df.columns[1:]
# for i,col in enumerate(cols1):
#     if col!=cols2[i]:
#         print(i,col,cols2[i])
# for i,col in enumerate(cols1):
#     if col!=cols2[i]:
#         print(i,col,cols2[i])

# shutil.copy(temp_database_name,main_database_name)
# df = pl.read_parquet("TestPleaseWork.parquet")
# print(df)

# Example data
# data = [
#     {"timestamp": "2023-02-20T19:52:16+00:00"},
#     {"timestamp": "2023-01-15T15:32:12+00:00"},
#     {"timestamp": "2023-02-28T08:21:34+00:00"},
#     {"timestamp": "2023-03-10T10:45:22+00:00"},
# ]
#
# # Get the current time
# now = datetime.datetime.now(datetime.timezone.utc)
#
# # Define the time delta for one month (approximate)
# one_month = datetime.timedelta(days=30)
#
# # Filter the data
# filtered_data = [
#     item for item in data
#     if now - datetime.datetime.fromisoformat(item["timestamp"]) > one_month
# ]
#
# print(filtered_data)

# data = Data()
# schema = data.main_df.schema
# null_columns = [key for key in schema if str(schema[key]) == 'Null']
# for col_name in null_columns:
#     data.main_df = data.main_df.with_column(data.main_df[col_name].cast(pl.Int64))
#
# data.main_df.write_parquet(temp_database_name)

# data=Data()
# print(data.main_df)
# print(data.main_df.schema)
# columns = list(data.main_df.schema.keys())
# print(list(columns))
# new_dict={'a':5,'b':6,'d':7}
# type_dict={'a': pl.Int8, 'b' : pl.Int8,'d':pl.Int8}
# type_dict2 = {'c':pl.Int8}
# type_dict3 = {**type_dict,**type_dict2}
# print(type_dict3)
# new_df = pl.DataFrame(new_dict,schema=type_dict)
# print(new_df)

# new_dict={'a' :1,'b':2,'c':3, 'e':'this','h':'works', }
# a=['a','b','c']
# b=[pl.Int8,pl.Int16,pl.Int32]
# schema_dict={'e' : pl.Utf8, 'h' : pl.Utf8} | {x:y for (x,y) in zip(a,b)}
# print(new_dict)
# new_df = pl.DataFrame(new_dict,schema=schema_dict)
# print(new_df)

#
# list1=['a','b','c']
# pl.Config.set_tbl_cols(15)
# df = pl.read_parquet("TestScoresDB.parquet")
# print(df,df.schema)
# Make both the normal dict(??) and the dtype dict in this way
# data=Data()
# titles = data.anime_df.columns[1:]
# main_df_columns = ['Username', 'Mean Score', 'Scored Shows'] + titles
# new_dict =


# new_dict['Username'] = ''
# for key in columns:
#     new_dict[key] =
# temp_df = pl.read_parquet(temp_database_name)
# schema=temp_df.schema
# for key in schema:
#     if schema[key] != schema['Steins;Gate']:
#         print(key, schema[key])

# titles = data.anime_df.columns
# rows = data.anime_df.rows()
# ids = rows[0][1:]
# scored_amount = rows[2][1:]
# k=5
# data = Data()
# cols = data.anime_df.columns[1:]
# print(len(cols))
# ids = [[i] for i in range(len(cols))]
# print(len(ids))
# indexes_df = pl.DataFrame(ids, schema=cols)
# print(indexes_df)
# print(pl.read_parquet("AnimeIndexes.parquet"))



# urls=[]
# headers=get_headers()
# urls.append('https://myanimelist.net/people/9834/Kouhei_Horikoshi')
# urls.append('https://myanimelist.net/profile/BaronBrixius')
# urls.append('https://myanimelist.net/anime/54789/Boku_no_Hero_Academia_7th_Season')
# urls.append('https://myanimelist.net/anime/5114/Fullmetal_Alchemist__Brotherhood')
# urls.append('https://myanimelist.net/anime/season/2009/spring')
# urls.append('https://myanimelist.net/anime/5680/K-On')
# urls.append(f'https://api.myanimelist.net/v2/users/' \
#           f'BaronBrixius/animelist?fields=list_status&limit=1000&sort=list_score&nsfw=True')

# test_dict={}
# df = pl.read_csv("AnimeDB.csv")
# df.write_parquet("AnimeDB.parquet")

# test_dict={}
# df = pl.read_parquet("TempScoresDBFinal.parquet")
# cols = [col for col in df.columns[4:] if col]
# for i,col in enumerate(cols):
#     test_dict[col] = [i]
# print(test_dict)
# table = pa.Table.from_pydict(test_dict)
# print(table)
# pq.write_table(table, 'AnimeIndexes.parquet')
#
# test = Data()
# print(test.main_df)
# print(test.anime_df)
# print(test.MAL_users_list[0:5])
# print(test.blacklist[0:5])
# print(test.anime_df)
# print(test.MAL_users_list[0:5])
# print(test.blacklist[0:5])
# print(responses[6]["data"])
# print(responses[4])
# print(5)
@timeit
def test_polars_load():
    polars_test = pl.read_parquet(main_database_name)

@timeit
def test_dusk_load():
    dask_test = dd.read_parquet(main_database_name)
    print(5)

@timeit
def test_pandas_load():
    pandas_test = pd.read_parquet(main_database_name)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10 ** 6:.2f} MB; Peak: {peak / 10 ** 6:.2f} MB")  # tracemall

@timeit
def test_soup_speed():
    url = 'https://myanimelist.net/profile/Neerou_'
    page_html = call_function_through_process(get_search_results, url)
    soup = BeautifulSoup(page_html.text, "html.parser")
    test = soup.findAll("span", {"class": "user-status-data di-ib fl-r"})
    date_str = test[1].text
    date_obj = datetime.datetime.strptime(date_str, "%b %d, %Y")

    # Add timezone information (replace tzinfo=None with tzinfo=timezone.utc if you are using Python 3.9+)
    date_obj = date_obj.replace(tzinfo=datetime.timezone.utc)

    formatted_date = date_obj.isoformat()
    print(formatted_date)
# headers = get_headers()
# print(fetch_concurrently(urls, 3, headers=headers))
# print(fetch_sequentially(urls,headers=headers))

# tracemalloc.start()
# test_polars_load()

# test_df = pl.read_parquet("TestDB1.parquet")
# # test_df2 = pl.read_parquet("ScoresDBFinal.parquet")
# print(test_df)
# data=Data()
# # print(data.main_df)
# # col = data.main_df.filter(pl.col("Kintamani Dog") == np.nan)
# data.main_df = data.main_df.fill_nan(None)
# # print(data.main_df)
# cols = data.main_df.columns[3:]
# print(cols)
# for col in cols:
#     data.main_df = data.main_df.with_columns(pl.col(col).cast(pl.UInt8, strict=True))
# print(data.main_df)
# data.main_df.write_parquet("TestDB1.parquet")



# if __name__=='__main__':
#     freeze_support()
#     df = pl.read_parquet("AnimeIndexes.parquet")
#     cols = df.columns
#     test_dict = {v: k for (k, v) in enumerate(cols)}
#     print(test_dict)



