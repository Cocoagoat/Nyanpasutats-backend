from filenames import *
from general_utils import *
from MAL_utils import *
from AnimeDB import AnimeDB
from Graphs import Graphs
from sortedcollections import OrderedSet


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
            cls._instance._entry_tags_dict = {}
            cls._instance._show_tags_dict = {}
            cls._instance._tags_per_category = {}
            cls._instance._anime_db = AnimeDB()
            # self._all_anilist_tags = OrderedSet()
            cls._instance._all_anilist_tags = []
            cls._instance._all_single_tags = []
            cls._instance._all_genres = []
            cls._instance._all_studios = []
            cls._instance._all_doubletags = []
            cls._instance._single_tags_used_for_doubles = []
            cls._instance.graphs = Graphs()
            cls._instance.tag_types = ["Single", "Double"]
        return cls._instance

    def __init__(self):
        # All properties are loaded on demand
        pass

    @property
    def entry_tags_dict(self):
        if not self._entry_tags_dict:
            try:
                print("Loading entry-tags dictionary")
                self._entry_tags_dict = load_pickled_file(entry_tags_filename)
                print("Entry-tags dictionary loaded successfully")
            except FileNotFoundError:
                print("Entry-tags dictionary not found. Creating new shows-tags dictionary")
                self.get_entry_tags()
        return self._entry_tags_dict

    @property
    def show_tags_dict(self):
        if not self._show_tags_dict:
            try:
                print("Loading shows-tags dictionary")
                self._show_tags_dict = load_pickled_file(shows_tags_filename)
                print("Shows-tags dictionary loaded successfully")
            except FileNotFoundError:
                print("Shows-tags dictionary not found. Creating new shows-tags dictionary")
                self.get_shows_tags()
        return self._show_tags_dict

    @property
    def all_anilist_tags(self):
        if not self._all_anilist_tags:
            self._all_anilist_tags = self.all_single_tags + self.all_genres + self.all_studios + self.all_doubletags
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
    def single_tags_used_for_doubles(self):
        pass

    @property
    def tags_per_category(self):
        if not self._tags_per_category:
            self._tags_per_category = {x['category']: [] for entry in self.entry_tags_dict.keys()
                                       for x in self.entry_tags_dict[entry]['Tags']}

            for entry in self.entry_tags_dict.keys():
                for tag in self.entry_tags_dict[entry]['Tags']:
                    if tag['name'] not in self._tags_per_category[tag['category']]:
                        self._tags_per_category[tag['category']].append(tag['name'])

            themes = [x for key, value in self._tags_per_category.items()
                      for x in value if key.startswith("Theme")]
            if "Episodic" in self._tags_per_category['Technical']:
                themes.append("Episodic")
            del self._tags_per_category['Technical']

            for key in list(self._tags_per_category.keys()):
                if key.startswith("Theme"):
                    del self._tags_per_category[key]

            split_themes = split_list_interval(themes, 10)

            doubles = self.get_full_doubletags_list(sorted=True)

            split_doubles = split_list_interval(doubles, 30)

            theme_dict = {f"Themes-{i + 1}": split_themes[i] for i in range(len(split_themes))}

            doubles_dict = {f"Doubles-{i + 1}": split_doubles[i] for i in range(len(split_doubles))}

            tag_counts = self.get_single_tag_counts()
            relevant_tags = []
            for tag, count in tag_counts.items():
                if count >= tag_counts['Thriller']:  # last genre
                    relevant_tags.append(tag)
                else:
                    break

            doubles_per_tag = {f"Doubles-{tag}" : [double_tag
                                                         for double_tag in doubles if f"<{tag}>"
                                                         in double_tag] for tag in relevant_tags}

            self._tags_per_category = self._tags_per_category | theme_dict | doubles_per_tag | doubles_dict
        return self._tags_per_category

    @property
    def relevant_tags_for_double(self):
        pass

    def get_full_tags_list(self):
        with open("NSFWTags.txt", 'r') as file:
            banned_tags = file.read().splitlines()
        tags = OrderedSet()
        for show, show_dict in self.entry_tags_dict.items():
            if 'Tags' in show_dict:
                for tag_dict in show_dict['Tags']:
                    if tag_dict['name'] not in banned_tags:
                        tags.add(tag_dict['name'])
        return list(tags)

    def get_single_tag_counts(self):
        """Goes over each show in the newly created entry_tags_dict to gather every tag that appears at least once,
        # as well as checks how often they appear."""
        tag_counts = {}
        banned_tags = self.get_banned_tags()
        for entry, entry_data in self._entry_tags_dict.items():
            tags_genres = self.gather_tags_and_genres(entry_data)
            for tag in tags_genres:  # Genres are strings, tags are dictionaries with ['name'] and ['percentage']
                tag_name = tag['name'] if type(tag) is dict else tag
                tag_p = tag['percentage'] if type(tag) is dict else 1
                if tag_name not in banned_tags:
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
        return list(genres)

    def get_full_studios_list(self):
        studios = OrderedSet()
        studio_dict={}
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

    def get_entry_tags(self):
        def get_recommended_shows():
            try:
                sorted_recs = sorted(entry['recommendations']['nodes'], key=lambda x: x['rating'], reverse=True)
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
                    print(rec_index, entry)
                # Can't take title directly from the recommendation object,
                # might be different from the MAL title we have in the database
                rec_dict[rec_MAL_title] = rec['rating']
            return rec_dict

        def get_shows_from_page(page_num):
            variables = {"page": page_num, "isAdult": False}
            response = requests.post(url, json={"query": self.query, "variables": variables}).json()
            show_list = response["data"]["Page"]["media"]
            has_next_page = response["data"]["Page"]["pageInfo"]["hasNextPage"]
            return show_list, has_next_page

        def get_entry_data(entry):
            try:
                index = ids.index(entry["idMal"])
            except (TypeError, ValueError):
                return
            title = anime_db.titles[index]  # Check this, might not be synchronized!
            # show_stats = anime_db.get_stats_of_shows([title], ["Episodes", "Duration"])
            show_recommendations = get_recommended_shows()

            if title in relevant_shows:
                try:
                    main_entry, _ = self.graphs.find_related_entries(title)
                except TypeError:
                    return

                entry_data = {
                    "Tags": [{"name": tag["name"], "percentage": self.adjust_tag_percentage(tag["rank"]),
                              "category": tag["category"]} for tag in entry["tags"]
                             if tag["rank"] >= 60 and tag['name'] not in banned_tags],
                    "Genres": entry["genres"],
                    "Studio": entry["studios"]["nodes"][0]["name"] if entry["studios"]["nodes"] else None,
                    "Recommended": show_recommendations,
                    "Main": main_entry
                    # If recs amount less than 25 start penalizing?
                }
                return title, entry_data

        anime_db = AnimeDB()
        with open("NSFWTags.txt", 'r') as file:
            # Banned tags are mostly NSFW stuff that doesn't exist in regular shows
            banned_tags = file.read().splitlines()
        url = "https://graphql.anilist.co"
        ids = anime_db.df.row(anime_db.stats['ID'])[1:]
        has_next_page = True
        page_num = 1
        relevant_shows = anime_db.partial_df.columns

        # First, we create entry_tags_dict. This dictionary has the tag-related details of each RELEVANT
        # entry (with relevance defined in anime_db.partial_df, basically not too short and not TOO obscure)
        while has_next_page:
            print(f"Currently on page {page_num}")
            entry_list, has_next_page = get_shows_from_page(page_num)
            for entry in entry_list:
                try:
                    title, entry_data = get_entry_data(entry)
                except TypeError:
                    continue
                self._entry_tags_dict[title] = entry_data
            page_num += 1
            time.sleep(1)

        # I have no idea wtf this is but every elegant mitigation measure against it failed
        del self._entry_tags_dict['Black Clover: Mahou Tei no Ken']

        # tag_counts = self.get_single_tag_counts()
        #
        # # If a tag appears less than 3.45 times (weighted by %) throughout each entry, we disregard it completely.
        # relevant_tags = {tag: count for tag,count in
        #               sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        #               if count >= 3.45}
        # all_single_tags = list(relevant_tags.keys())

        relevant_double_tag_counts = self.get_double_tag_counts(only_relevant=True)
        relevant_double_tags = list(relevant_double_tag_counts.keys())


        # If a double tag appears less than 15 times (weighted by %) throughout each entry,
        # we disregard it completely. Since there are ~50k double tags we only want the most important ~2k tags,
        # otherwise it'll be cluttered with tags that almost never appear together + take ages to run.
        # relevant_double_tags = list({tag: count for tag, count in double_tag_counts.items() if count >= 15}.keys())

        # Add the double tags to the data dict of each entry
        for entry, entry_data in self._entry_tags_dict.items():
            tags_genres = self.gather_tags_and_genres(entry_data)
            self._entry_tags_dict[entry]['DoubleTags'] = []
            for i, tag1 in enumerate(tags_genres):
                tag1_name = tag1['name'] if type(tag1) is dict else tag1
                tag1_p = tag1['percentage'] if type(tag1) is dict else 1
                for j in range(i+1, len(tags_genres)):
                    tag2 = tags_genres[j]
                    tag2_name = tag2['name'] if type(tag2) is dict else tag2
                    tag2_p = tag2['percentage'] if type(tag2) is dict else 1
                    if f"<{tag1_name}>x<{tag2_name}>" in relevant_double_tags:
                        self._entry_tags_dict[entry]['DoubleTags'].append(
                            {'name' : f"<{tag1_name}>x<{tag2_name}>", 'percentage' : min(tag1_p, tag2_p)})
                    elif f"<{tag2_name}>x<{tag1_name}>" in relevant_double_tags:
                        self._entry_tags_dict[entry]['DoubleTags'].append(
                            {'name': f"<{tag2_name}>x<{tag1_name}>", 'percentage': min(tag1_p, tag2_p)})

        save_pickled_file(entry_tags_filename, self._entry_tags_dict)

    def get_shows_tags(self):
        def calc_length_coeff(entry, stats):
            return round(min(1, stats[entry]["Episodes"] *
                       stats[entry]["Duration"] / 200),3)

        def get_tags_from_entries(double=False):
            show_tags={}
            if not double:
                all_entries_tags_list = [{'name': tag['name'], 'percentage': tag['percentage']}
                                         for entry in show_related_entries
                                         for tag in self.entry_tags_dict[entry]['Tags']]
            else:
                all_entries_tags_list = [{'name': double_tag['name'], 'percentage': double_tag['percentage']}
                                         for entry in show_related_entries
                                         for double_tag in self.entry_tags_dict[entry]['DoubleTags']]
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
                entry_recommended = self.entry_tags_dict[entry]['Recommended']
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


        processed_titles = []
        show_amount = len(self.entry_tags_dict.keys())

        for i, entry in enumerate(self.entry_tags_dict.keys()):

            # print(show)
            if i % 100 == 0:
                print(f"Currently on show {i} of {show_amount}")
                # raise ValueError

            if entry in processed_titles:
                continue

            try:
                main_entry, show_related_entries = self.graphs.find_related_entries(entry)
            except TypeError:
                continue  # Some entries exist on MAL, but not on Anilist (and thus they do not appear
            # in the tags dictionary which is built using Anilist's tags). Those entries
            # are all either recaps, commercials or other side entries that we normally
            # would not want to analyze either way.

            show_related_entries = [entry for entry in show_related_entries
                                    if entry in self.entry_tags_dict.keys()]

            #
            # all_entries_tags_list = [{tag['name']: tag['percentage']} for entry in show_related_entries
            #                          for tag in self.entry_tags_dict[entry]['Tags']]

            # length_coeff = calc_length_coeff(entry, stats_of_related_entries)
            self._show_tags_dict[main_entry] = {}

            # Tags (get the tags of all entries, if several entries have the same tag at different
            # percentages then take the highest percentage among them as the show's tag percentage)
            show_tags = get_tags_from_entries(double=False)
            self._show_tags_dict[main_entry]['Tags'] = [{'name': tag_name, 'percentage': tag_p}
                                                        for tag_name, tag_p in show_tags.items()]

            show_double_tags = get_tags_from_entries(double=True)
            self._show_tags_dict[main_entry]['DoubleTags'] = [{'name': tag_name, 'percentage': tag_p}
                                                        for tag_name, tag_p in show_double_tags.items()]
            # for tag in all_entries_tags_list:
            #     for key, value in tag.items():
            #         if key not in show_tags or value > show_tags[key]:
            #             # (each tag a show has gets the maximum value of said tag among its entries)
            #             show_tags[key] = value
            # show_tags = {tag[0]: tag[1] for tag in sorted(show_tags.items(),
            #                                               key=lambda x: x[1], reverse=True)}

            # for tag in all_entries_tags_list:
            #     if tag['name'] not in show_tags or tag['percentage'] > show_tags[tag['name']]:
            #         # Each tag a show has gets the maximum value of said tag among its entries
            #         show_tags[tag['name']] = tag['percentage']
            # show_tags = {tag[0]: tag[1] for tag in sorted(show_tags.items(),
            #                                               key=lambda x: x[1], reverse=True)}


            # for name, percentage in show_tags.items():
            #     self._show_tags_dict[main_entry]['Tags'].append({'name': name, 'percentage': percentage})

            # Genres (simply add all the unique genres of all the entries)
            show_genres = list(set([genre for entry in show_related_entries
                                    for genre in self.entry_tags_dict[entry]['Genres']]))
            self._show_tags_dict[main_entry]['Genres'] = show_genres

            # Studios (same as genres)
            show_studios = {entry: self.entry_tags_dict[entry]['Studio'] for entry in show_related_entries}
            self._show_tags_dict[main_entry]['Studios'] = show_studios

            # Recommended (we take the recommendations for each entry the show has, sum up the values,
            # and take the top 5 with the highest recommendation values).
            show_recommended = get_recommended_shows()
            # for entry in show_related_entries:
            #     entry_recommended = self.entry_tags_dict[entry]['Recommended']
            #     for rec_entry, rec_value in entry_recommended.items():
            #         if rec_entry not in show_recommended:
            #             show_recommended[rec_entry] = rec_value
            #         else:
            #             show_recommended[rec_entry] += rec_value

            self._show_tags_dict[main_entry]['Recommended'] = show_recommended

            # Related
            stats_of_related_entries = self.anime_db.get_stats_of_shows(show_related_entries,
                                                                        ["Episodes", "Duration"])
            related = {}
            for entry in show_related_entries:
                length_coeff = calc_length_coeff(entry, stats_of_related_entries)
                related[entry] = length_coeff
            self._show_tags_dict[main_entry]['Related'] = related

            # Add main show to entry_dict
            for entry in show_related_entries:
                self._entry_tags_dict[entry]['Main'] = main_entry

            # # Doubles
            # show_double_tags=[]
            # all_entries_double_tags_list = [{tag['name']: tag['percentage']} for entry in show_related_entries
            #                                 for double_tag in self.entry_tags_dict[entry]['Doubles']]
            # for tag in all_entries_double_tags_list:
            #     if tag['name'] not in show_tags or tag['percentage'] > show_tags[tag['name']]:
            #         # Each tag a show has gets the maximum value of said tag among its entries
            #         show_double_tags[tag['name']] = tag['percentage']
            # show_double_tags = {tag[0]: tag[1] for tag in sorted(show_double_tags.items(),
            #                                               key=lambda x: x[1], reverse=True)}
            #
            # self._show_tags_dict[main_entry]['Doubles'] = [{'name': tag_name, 'percentage': tag_p}
            #                                                for tag_name, tag_p in show_double_tags.items()]

            # To avoid repeat processing
            processed_titles = processed_titles + show_related_entries

        save_pickled_file(shows_tags_filename, self._show_tags_dict)
        # save_pickled_file(entry_tags_filename, self._entry_tags_dict)


    def shows_per_tag(self):
        self.shows_per_tag = {tag : [] for tag in self.all_anilist_tags}
        for tag_name in self.all_anilist_tags:
            for entry, entry_data in self.entry_tags_dict.items():
                entry_tags = entry_data['Tags'] + entry_data['DoubleTags'] + entry_data['Genres'] + [entry_data['Studio']]
                for tag in entry_tags:
                    try:
                        tag = tag['name']
                    except KeyError:
                        pass
                    if tag == tag_name:
                        self.shows_per_tag[tag_name].append(entry)
                        break



    def get_shows_tags_pairs(self, entry):
        show_tags = self.show_tags_dict[entry]['Tags']
        for i, tag1 in enumerate(show_tags):
            for j in range(i+1, len(show_tags)):
                pass

    @staticmethod
    def get_tag_index(tag_name, show_tags_list):
        for i, tag in enumerate(show_tags_list):
            if tag_name == tag['name']:
                return i
        return -1

    def get_avg_score_of_show(self,show, mean_score_per_show):
        entry_count = 1
        avg = mean_score_per_show[show].item()
        for entry, length_coeff in self.show_tags_dict[show]['Related'].items():
            if length_coeff == 1 and entry != show:
                avg += mean_score_per_show[entry].item()
                entry_count += 1
        return round(avg/entry_count,2)

    def get_max_score_of_show(self,show,mean_score_per_show):
        max_score = mean_score_per_show[show].item()
        for entry, length_coeff in self.show_tags_dict[show]['Related'].items():
            if length_coeff == 1 and entry != show:
                try:
                    max_score = max(mean_score_per_show[entry].item(),max_score)
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
        banned_tags = self.get_banned_tags()
        if 'Tags' in show_dict:
            tags_genres.extend([tag for tag in show_dict['Tags'] if tag['name'] not in banned_tags])
        if 'Genres' in show_dict:
            tags_genres.extend(show_dict['Genres'])
        return tags_genres

    @staticmethod
    def get_banned_tags():
        with open("NSFWTags.txt", 'r') as file:
            banned_tags = file.read().splitlines()
        return banned_tags



