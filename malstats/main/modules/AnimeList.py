from abc import ABC, abstractmethod
from main.modules.AnimeListFormatter import ListFormatter
from main.models import AnimeData
from django.db.models import Case, When

from main.modules.general_utils import redis_cache_wrapper


class AnimeList(ABC):
    """Container class for anime lists, currently supports only MAL and Anilist.
    Contains both the original object (list_obj) returned from MyAnimeList/Anilist (slightly
    modified in case of the latter and the unified format (Anime) for both lists (list).

    """
    def __init__(self, list_obj):
        self.list_obj = list_obj
        self._list = None
        self._all_titles = None
        self._all_ids = None

    # @redis_cache_wrapper(timeout=5 * 60)
    @property
    def list(self):
        if not self._list:
            self._list = ListFormatter(self).formatted_list
        return self._list

    @property
    def all_titles(self):
        if not self._all_titles:
            all_titles_and_ids = self._get_all_list_titles_and_ids()
            self._all_titles = all_titles_and_ids['titles']
            self._all_ids = all_titles_and_ids['ids']
        return self._all_titles

    @property
    def all_ids(self):
        if not self._all_ids:
            all_titles_and_ids = self._get_all_list_titles_and_ids()
            self._all_titles = all_titles_and_ids['titles']
            self._all_ids = all_titles_and_ids['ids']
        return self._all_ids

    @abstractmethod
    def _get_all_list_titles_and_ids(self):
        pass

    @abstractmethod
    def get_entry_from_id(self):
        pass

    iter_flag = False

    def __iter__(self):
        if self.iter_flag:
            return iter(self.list_obj)
        if not self._list:
            self.iter_flag = True
            _ = self.list
            self.iter_flag = False
        return iter(self.list)

    def __getitem__(self, key):
        return self.list[key]


class MALList(AnimeList):
    def __init__(self, list_obj):
        super().__init__(list_obj)

    def _get_all_list_titles_and_ids(self):
        titles = [anime['node']['title'] for anime in self.list_obj]
        ids = [anime['node']['id'] for anime in self.list_obj]
        return {'ids': ids, 'titles': titles}

    def get_entry_from_id(self, id):
        for anime in self.list_obj:
            if id == anime['node']['id']:
                return anime
        return None


class AniList(AnimeList):
    def __init__(self, list_obj):
        super().__init__(list_obj)

    def _get_all_list_titles_and_ids(self):
        ids = [anime['media']['idMal'] for anime in self.list_obj]
        relevant_shows = AnimeData.objects.filter(mal_id__in=ids)
        preserved_order = Case(*[When(mal_id=id, then=pos) for pos, id in enumerate(ids)])
        relevant_shows_ordered = relevant_shows.order_by(preserved_order)
        if len(ids) != len(relevant_shows_ordered):
            relevant_shows_dict = {obj.mal_id: obj for obj in list(relevant_shows_ordered)}
            relevant_shows_ordered = [relevant_shows_dict.get(id) for id in ids]
        return {'ids': ids, 'titles': [anime.name if anime else None for anime in relevant_shows_ordered]}

    def get_entry_from_id(self, id):
        for anime in self.list_obj:
            if id == anime['media']['idMal']:
                return anime
        return None
