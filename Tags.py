from filenames import *
from general_utils import *
from MAL_utils import *
from AnimeDB import AnimeDB
from Graphs import Graphs


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
        self._shows_tags_dict2 = {}
        self.anime_db = AnimeDB()
        self._all_tags_list = OrderedSet()
        self.graphs = Graphs()

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

    @property
    def shows_tags_dict2(self):
        if not self._shows_tags_dict2:
            try:
                print("Loading shows-tags2 dictionary")
                self._shows_tags_dict2 = load_pickled_file(shows_tags_filename2)
                print("Shows-tags2 dictionary loaded successfully")
            except FileNotFoundError:
                print("Shows-tags2 dictionary not found. Creating new shows-tags dictionary")
                self.get_shows_tags2()
        return self._shows_tags_dict2

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

    def get_shows_tags2(self):
        def calc_length_coeff(entry, stats):
            return round(min(1, stats[entry]["Episodes"] *
                       stats[entry]["Duration"] / 200),3)

        processed_titles = []

        for i, show in enumerate(self.shows_tags_dict.keys()):

            show_amount = len(self.shows_tags_dict.keys())
            # print(show)
            if i%100==0:
                print(f"Currently on show {i} of {show_amount}")

            if show in processed_titles:
                continue

            try:
                main_entry, show_related_entries = self.graphs.find_related_entries(show)
            except TypeError:
                continue

            # Some entries exist on MAL, but not on Anilist (and thus they do not appear
            # in the tags dictionary which is built using Anilist's tags). Those entries
            # are all either recaps, commercials or other side entries that we normally
            # would not want to analyze either way.
            show_related_entries = [entry for entry in show_related_entries
                                    if entry in self.shows_tags_dict.keys()]

            all_entries_tags_list = [{tag['name']: tag['percentage']} for entry in show_related_entries
                                     for tag in self.shows_tags_dict[entry]['Tags']]

            # length_coeff = calc_length_coeff(entry, stats_of_related_entries)
            self._shows_tags_dict2[main_entry] = {}

            # Tags
            show_tags = {}
            for d in all_entries_tags_list:
                for key, value in d.items():
                    if key not in show_tags or value > show_tags[key]:
                        show_tags[key] = value
            show_tags = {tag[0]: tag[1] for tag in sorted(show_tags.items(),
                                                          key=lambda x: x[1], reverse=True)}

            self._shows_tags_dict2[main_entry]['Tags'] = []
            for name, percentage in show_tags.items():
                self._shows_tags_dict2[main_entry]['Tags'].append({'name': name, 'percentage': percentage})

            # Genres
            show_genres = list(set([genre for entry in show_related_entries
                                    for genre in self.shows_tags_dict[entry]['Genres']]))
            self._shows_tags_dict2[main_entry]['Genres'] = show_genres

            # Studios
            show_studios = {entry: self.shows_tags_dict[entry]['Studio'] for entry in show_related_entries}
            self._shows_tags_dict2[main_entry]['Studios'] = show_studios

            # Recommended
            show_recommended={}
            for entry in show_related_entries:
                entry_recommended = self.shows_tags_dict[entry]['Recommended']
                for key, value in entry_recommended.items():
                    if key not in show_recommended:
                        show_recommended[key] = value
                    else:
                        show_recommended[key] += value

            sorted_recommended_shows = sorted(show_recommended.items(),
                                              reverse=True, key=lambda x: x[1])[0:5]
            show_recommended = {rec[0]: rec[1] for rec in sorted(sorted_recommended_shows,
                                                                 key=lambda x: x[1], reverse=True)}
            self._shows_tags_dict2[main_entry]['Recommended'] = show_recommended

            # Related
            stats_of_related_entries = self.anime_db.get_stats_of_shows(show_related_entries,
                                                                        ["Episodes", "Duration"])
            related = {}
            for entry in show_related_entries:
                length_coeff = calc_length_coeff(entry, stats_of_related_entries)
                related[entry] = length_coeff
            self._shows_tags_dict2[main_entry]['Related'] = related

            # To avoid repeat processing
            processed_titles = processed_titles + show_related_entries

        save_pickled_file(shows_tags_filename2, self._shows_tags_dict2)

    @staticmethod
    def get_tag_index(tag_name, show_tags_list):
        for i, tag in enumerate(show_tags_list):
            if tag_name == tag['name']:
                return i
        return -1

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
