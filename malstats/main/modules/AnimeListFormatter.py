from abc import ABC, abstractmethod
from main.modules.Tags import Tags
from main.modules.MAL_utils import MALUtils
# from main.modules.AnimeListHandler import AnilistHandler, AnimeListHandler, MALListHandler
from main.modules.AnimeEntryHandler import AnimeEntryHandler, MALEntryHandler, AnilistEntryHandler
from main.modules.general_utils import timeit, redis_cache_wrapper


# Might refactor into strategy pattern in the future, but for now it's much easier
# to use this way


class AnimeListFormatter(ABC):
    """Formats an anime list object (as returned from MAL or Anilist) directly into a dictionary
    of the following format : {title_1 : {'stat_1' : stat_1_value,
    'stat_2' : stat_2_value , ... , 'stat_n' : stat_n_value} """
    def __init__(self, anime_list, stats_to_get, include_sequels=True, include_dropped=True, include_unscored=False):
        self.anime_list = anime_list
        self.include_sequels = include_sequels
        self.include_dropped = include_dropped
        self.include_unscored = include_unscored
        self.allowed_stats = ["score", "status", "num_watched"]
        self._formatted_list = {}
        self._meets_conditions_dict = {}  # Exists so we won't have to check conditions every time
        self.tags = Tags()
        if not all([stat in self.allowed_stats for stat in stats_to_get]):
            raise ValueError(f"Only the following stats can be included in the"
                             f" formatted object : {self.allowed_stats}")
        self.stats_to_get = stats_to_get

    @property
    @abstractmethod
    def formatted_list(self):
        pass

    @abstractmethod
    def _anime_meets_conditions(self,anime):
        pass

    @abstractmethod
    def _add_score(self):
        pass

    @abstractmethod
    def _add_list_status(self):
        pass

    @abstractmethod
    def _add_num_watched(self):
        pass


class AnimeListConditionChecker(ABC):

    @staticmethod
    @abstractmethod
    def is_sequel(anime):
        pass

    @staticmethod
    @abstractmethod
    def is_unscored(anime):
        pass

    @staticmethod
    @abstractmethod
    def is_dropped(anime):
        pass

    @staticmethod
    @abstractmethod
    def is_ptw(anime):
        pass


# class ListConditionChecker():
#
#     def __init__(self, anime_entry_handler: AnimeEntryHandler, include_sequels=True, include_dropped=True,
#                  include_unscored=False, custom_condition_func=None):
#         self.anime_entry_handler = anime_entry_handler
#         self.condition_func = self.create_condition_func(
#             include_sequels, include_dropped, include_unscored
#         ) if not custom_condition_func else custom_condition_func
#
#     def is_sequel(self, entry):
#         tags = Tags()  # Tags is a Singleton and all its properties are on-demand so it'll only be instantiated once
#         try:
#             title = self.anime_entry_handler.get_title_from_entry(entry)
#         except ValueError:
#             return True  # True if it's a sequel, so we exclude it if needed
#         return title not in tags.show_tags_dict.keys()
#
#     def is_unscored(self,entry):
#         score = self.anime_entry_handler.get_score_from_entry(entry)
#         return not bool(score)
#
#     def is_dropped(self, entry):
#         list_status = self.anime_entry_handler.get_list_status_from_entry(entry)
#         return list_status.lower() == 'dropped'
#         # Anilist returns DROPPED, MAL returns dropped
#
#     def is_ptw(self, entry):
#         list_status = self.anime_entry_handler.get_list_status_from_entry(entry)
#         return list_status == 'plan_to_watch' or list_status == "PLANNING"
#
#     def create_condition_func(self, include_sequels, include_dropped, include_unscored):
#         def condition_func(entry):
#             return (not self.is_ptw(entry)
#              and (include_sequels or not self.is_sequel(entry))
#              and (include_dropped or not self.is_dropped(entry))
#              and (include_unscored or not self.is_unscored(entry)))
#         return condition_func
#
#     def anime_meets_conditions(self, entry):
#         return self.condition_func(entry)


class ListFormatter():
    def __init__(self, anime_list, stats_to_get=None):
        site = AnimeEntryHandler.determine_list_site(anime_list)
        self.anime_entry_handler = AnimeEntryHandler.get_concrete_handler(site)
        # self.condition_checker = ListConditionChecker(self.anime_list_handler, include_sequels,
        #          include_dropped, include_unscored, custom_condition_func)

        self.anime_list = anime_list
        # self.include_sequels = include_sequels
        # self.include_dropped = include_dropped
        # self.include_unscored = include_unscored

        self._formatted_list = {}
        # self._meets_conditions_cache = {}  # Exists so we won't have to check conditions every time
        # self.tags = Tags()
        # self.stats_to_get = stats_to_get
        all_stats = ["score", "list_status", "num_watched", "id", "updated_at"]
        self.stats_to_get = stats_to_get if stats_to_get else all_stats
        # if not all([stat in self.allowed_stats for stat in stats_to_get]):
        #     raise ValueError(f"Only the following stats can be included in the"
        #                      f" formatted object : {self.allowed_stats}")
        # self._stats_to_getter_mapping = {"score": self.anime_entry_handler.get_score_from_entry,
        #                                  "status": self.anime_entry_handler.get_list_status_from_entry,
        #                                  "num_watched": self.anime_entry_handler.get_num_watched_from_entry,
        #                                  "id": self.anime_entry_handler.get_id_from_entry,
        #                                  "updated_at" : self.anime_entry_handler.get_}

    @property
    def formatted_list(self):
        if not self._formatted_list:
            self._initialize_formatted_list()
            self._formatted_list = self._format_list()
        return self._formatted_list

    def _initialize_formatted_list(self):
        for entry in self.anime_list.list_obj:
            try:
                title = self.anime_entry_handler.get("title", entry)
                self._formatted_list = self._formatted_list | {title: {}}
            except ValueError:
                continue

    def _format_list(self):
        # This roundabout way is used to speed up computation - if we simply loop over the
        # entries we already have and get the stats from that, we'd have to get the MAL title
        # of each Anilist entry which will send a request to the SQLite database every time.
        # This actually makes the formatting take over a second for very large lists.
        all_titles = self.anime_list.all_titles
        all_ids = self.anime_list.all_ids

        for title, id in list(zip(all_titles, all_ids)):
            if not title:
                continue
            entry = self.anime_list.get_entry_from_id(id)
            for stat in self.stats_to_get:
                self._formatted_list[title] = self._formatted_list[title] | {
                    stat: self.anime_entry_handler.get(stat, entry)}

        return self._formatted_list


        # for entry in self.anime_list:
        #     try:
        #         title = self.anime_entry_handler.get_title_from_entry(entry)
        #     except ValueError:
        #         continue
        #     for stat in self.stats_to_get:
        #         self.formatted_list[title] = self.formatted_list[title] | {
        #             stat: self.anime_entry_handler.get(stat, entry)}
                # self._add_stat(stat, title, entry)

    # def _add_stat(self, stat_name, title, entry):
    #     get_stat = self._stats_to_getter_mapping[stat_name]
    #     self.formatted_list[title] = self.formatted_list[title] | {
    #         stat_name: get_stat(entry)}

    # def _add_score(self, title, entry):  # change to loop over all stats at once
    #     # for entry in self.anime_list.list_obj:
    #     #     title = self.anime_list_handler.get_title_from_entry(entry)
    #     #     if not self._meets_conditions_dict[title]:
    #     #         continue
    #     self.formatted_list[title] = self.formatted_list[title] | {
    #         'score': self.anime_list_handler.get_score_from_entry(entry)}
    #
    # def _add_list_status(self):
    #     for entry in self.anime_list.list_obj:
    #         title = self.anime_list_handler.get_title_from_entry(entry)
    #         if not self._meets_conditions_dict[title]:
    #             continue
    #         self.formatted_list[title] = self.formatted_list[title] | {
    #             'status': self.anime_list_handler.get_list_status_from_entry(entry)}
    #
    # def _add_num_watched(self):
    #     for entry in self.anime_list.list_obj:
    #         title = self.anime_list_handler.get_title_from_entry(entry)
    #         if not self._meets_conditions_dict[title]:
    #             continue
    #         self.formatted_list[title] = self.formatted_list[title] | {
    #             'num_watched': self.anime_list_handler.get_num_watched_from_entry(entry)}


# class MALListFormatter(AnimeListFormatter):
#
#     def __init__(self, anime_list, stats_to_get, include_sequels=True, include_dropped=True, include_unscored=False,
#                  ):
#         super().__init__(anime_list, stats_to_get, include_sequels, include_dropped, include_unscored)
#         self.condition_checker = ListConditionChecker(MALListHandler)
#         self._stats_to_function_mapping = {"score": self._add_score, "status": self._add_list_status,
#                                            "num_watched": self._add_num_watched}
#
#     @property
#     def formatted_list(self):
#         if not self._formatted_list:
#             self._initialize_formatted_list()
#             self._format_list()
#         return self._formatted_list
#
#     def _initialize_formatted_list(self):
#         self._formatted_list = {anime['node']['title']: {}
#                                 for anime in self.anime_list
#                                 if self._anime_meets_conditions(anime)}
#
#     def _anime_meets_conditions(self, anime):
#         meets_conditions = (not self.condition_checker.is_ptw(anime)
#                             and (self.include_sequels or not self.condition_checker.is_sequel(anime))
#                             and (self.include_dropped or not self.condition_checker.is_dropped(anime))
#                             and (self.include_unscored or not self.condition_checker.is_unscored(anime)))
#         self._meets_conditions_dict = self._meets_conditions_dict | {anime['node']['title']: meets_conditions}
#         return meets_conditions
#
#     def _format_list(self):
#         for stat in self.stats_to_get:
#             add_stat = self._stats_to_function_mapping[stat]
#             add_stat()
#
#     def _add_score(self):
#         for anime in self.anime_list:
#             title = anime['node']['title']
#             if not self._meets_conditions_dict[title]:
#                 continue
#             self.formatted_list[title] = self.formatted_list[title] | {'score': anime['list_status']['score']}
#
#     def _add_list_status(self):
#         for anime in self.anime_list:
#             title = anime['node']['title']
#             if not self._meets_conditions_dict[title]:
#                 continue
#             self.formatted_list[title] = self.formatted_list[title] | {'status': anime['list_status']['status']}
#
#     def _add_num_watched(self):
#         for anime in self.anime_list:
#             title = anime['node']['title']
#             if not self._meets_conditions_dict[title]:
#                 continue
#             self.formatted_list[title] = self.formatted_list[title] | {'num_watched': anime['list_status']['num_watched']}
#
#
# # class MALListConditionChecker(AnimeListConditionChecker):
# #
# #     @staticmethod
# #     def is_sequel(anime):
# #         tags = Tags()  # Tags is a Singleton and all its properties are on-demand so it'll only be instantiated once
# #         return anime['node']['title'] not in tags.show_tags_dict.keys()
# #
# #     @staticmethod
# #     def is_unscored(anime):
# #         return not bool(anime['list_status']['score'])
# #
# #     @staticmethod
# #     def is_dropped(anime):
# #         return anime['list_status']['status'] == 'dropped'
# #
# #     @staticmethod
# #     def is_ptw(anime):
# #         return anime['list_status']['status'] == 'plan_to_watch'
#
#
# class AnilistFormatter(AnimeListFormatter):
#     def __init__(self, anime_list, stats_to_get, include_sequels=True, include_dropped=True, include_unscored=False):
#         super().__init__(anime_list, stats_to_get, include_sequels, include_dropped, include_unscored)
#         self.condition_checker = ListConditionChecker(AnilistHandler)
#         self._stats_to_function_mapping = {"score": self._add_score, "status": self._add_list_status,
#                                            "num_watched": self._add_num_watched}
#
#     @property
#     def formatted_list(self):
#         if not self._formatted_list:
#             self._initialize_formatted_list()
#             self._format_list()
#         return self._formatted_list
#
#     def _initialize_formatted_list(self):
#
#         for lst in self.anime_list['lists']:
#             if lst['name'] == 'Planning':
#                 continue
#             for entry in lst['entries']:
#                 if self._anime_meets_conditions(entry):
#                     try:
#                         mal_title = AnilistHandler.get_title_from_entry(entry)
#                     except ValueError:
#                         continue
#                     self._formatted_list | {mal_title: {}}
#
#     def _anime_meets_conditions(self, anime):
#         meets_conditions = ((self.include_sequels or not self.condition_checker.is_sequel(anime))
#                             and (self.include_dropped or not self.condition_checker.is_dropped(anime))
#                             and (self.include_unscored or not self.condition_checker.is_unscored(anime)))
#         try:
#             mal_title = AnilistHandler.get_title_from_entry(anime)
#         except ValueError:
#             return False  # This shouldn't happen in theory but just as a failsafe
#         self._meets_conditions_dict = self._meets_conditions_dict | {mal_title : meets_conditions}
#         return meets_conditions
#
#     def _format_list(self):
#         for stat in self.stats_to_get:
#             add_stat = self._stats_to_function_mapping[stat]
#             add_stat()
#
#     def _add_score(self):
#         for anime in self.anime_list:
#             title = anime['node']['title']
#             if not self._meets_conditions_dict[title]:
#                 continue
#             self.formatted_list[title] = self.formatted_list[title] | {'score': anime['list_status']['score']}
#
#     def _add_list_status(self):
#         for anime in self.anime_list:
#             title = anime['node']['title']
#             if not self._meets_conditions_dict[title]:
#                 continue
#             self.formatted_list[title] = self.formatted_list[title] | {'status': anime['list_status']['status']}
#
#     def _add_num_watched(self):
#         for anime in self.anime_list:
#             title = anime['node']['title']
#             if not self._meets_conditions_dict[title]:
#                 continue
#             self.formatted_list[title] = self.formatted_list[title] | {
#                 'num_watched': anime['list_status']['num_watched']}


# class AnilistConditionChecker(AnimeListConditionChecker):
#
#     @staticmethod
#     def is_sequel(anime):
#         tags = Tags() # Tags is a Singleton and all its properties are on-demand so it'll only be instantiated once
#         mal_title = AnilistHandler.get_title_from_entry(anime)
#         return mal_title not in tags.show_tags_dict.keys()
#
#     @staticmethod
#     def is_unscored(anime):
#         return not bool(anime['score'])
#
#     @staticmethod
#     def is_dropped(anime):
#         return anime['list_status']['status'] == 'dropped'
#
#     @staticmethod
#     def is_ptw(anime):
#         return anime['list_status']['status'] == 'plan_to_watch'





