from filenames import *
from general_utils import *
from MAL_utils import *
from AnimeDB import AnimeDB


class Tags:
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
                self._shows_tags_dict = load_pickled_file(shows_tags_filename)
                print("Shows-tags dictionary loaded successfully")
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
                        studio_dict[show_dict['Studio']] += 1

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
                    rec_MAL_title = anime_db.titles[rec_index]
                except IndexError:
                    print(rec_index, media)
                # Can't take title directly from the recommendation object,
                # might be different from the MAL title we have in the database
                rec_dict[rec_MAL_title] = rec['rating']
            return rec_dict

        anime_db = AnimeDB()
        url = "https://graphql.anilist.co"
        ids = anime_db.df.row(anime_db.stats['ID'])[1:]
        has_next_page = True
        page = 1
        relevant_shows = anime_db.partial_df.columns

        while has_next_page:
            print(f"Currently on page {page}")
            variables = {"page": page, "isAdult": False}
            response = requests.post(url, json={"query": self.query, "variables": variables})
            response = response.json()
            media_list = response["data"]["Page"]["media"]
            for media in media_list:
                try:
                    index = ids.index(media["idMal"])
                except (TypeError, ValueError):
                    continue
                title = anime_db.titles[index]  # Check this, might not be synchronized!
                # show_stats = anime_db.get_stats_of_shows([title], ["Episodes", "Duration"])
                show_recommendations = get_recommended_shows()

                if title in relevant_shows:
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

        save_pickled_file(shows_tags_filename, self._shows_tags_dict)

    @staticmethod
    def get_tag_index(tag_name, show_tags_list):
        for i, tag in enumerate(show_tags_list):
            if tag_name == tag['name']:
                return i
        return -1

    @staticmethod
    def adjust_tag_percentage(tag):
        p = tag['percentage']
        if p >= 85:
            adjusted_p = round((tag['percentage'] / 100), 3)
        elif p >= 70:
            second_coeff = 20 / 3 - tag['percentage'] / 15  # Can be a number from 1 to 1.5
            adjusted_p = round((tag['percentage'] / 100) ** second_coeff, 3)
            # 85 ---> 1, 70 ---> 2, 15 forward means -1 so m=-1/15, y-2 = -1/15(x-70) --> y=-1/15x + 20/3
        elif p >= 60:
            second_coeff = 16 - tag['percentage'] / 5
            adjusted_p = round((tag['percentage'] / 100) ** second_coeff, 3)
        else:
            adjusted_p = 0
        return adjusted_p

    @staticmethod
    def load_tags_database():
        print(f"Unpacking part 1 of database")
        tag_df = pl.read_parquet(f"{tags_db_filename}-P1.parquet")
        for i in range(2, 1000):
            try:
                print(f"Unpacking part {i} of database")
                temp_df = pl.read_parquet(f"{tags_db_filename}-P{i}.parquet")
                tag_df = tag_df.vstack(temp_df)
            except FileNotFoundError:
                break
        return tag_df
