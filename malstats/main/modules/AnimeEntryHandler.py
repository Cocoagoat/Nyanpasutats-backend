from abc import ABC, abstractmethod
from main.modules.MAL_utils import MALUtils

from datetime import datetime


class AnimeEntryHandler(ABC):
    STAT_NAMES = ["score", "title", "list_status", "num_watched", "id", "updated_at"]

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def get_title_from_entry(entry):
        pass

    @staticmethod
    @abstractmethod
    def get_score_from_entry(entry):
        pass

    @staticmethod
    @abstractmethod
    def get_list_status_from_entry(entry):
        pass

    @staticmethod
    @abstractmethod
    def get_num_watched_from_entry(entry):
        pass

    @staticmethod
    @abstractmethod
    def get_id_from_entry(entry):
        pass

    @staticmethod
    @abstractmethod
    def get_updated_at_from_entry(entry):
        pass

    @staticmethod
    def determine_list_site(anime_list):
        from main.modules.AnimeList import MALList, AniList
        if type(anime_list) == MALList:
            return "MAL"
        elif type(anime_list) == AniList:
            return "Anilist"
        else:
            raise ValueError("Unknown list type, only MAL and Anilist are supported.")

    @staticmethod
    def get_concrete_handler(type):
        if type == 'MAL':
            return MALEntryHandler
        elif type == 'Anilist':
            return AnilistEntryHandler
        else:
            raise ValueError("Unknown website, only MAL and Anilist are supported.")

    @classmethod
    def get(cls, stat_name, entry):
        if stat_name not in cls.STAT_NAMES:
            raise ValueError(f"{stat_name} is not a valid seasonal stat")
        if stat_name == "score":
            return cls.get_score_from_entry(entry)
        elif stat_name == "title":
            return cls.get_title_from_entry(entry)
        elif stat_name == "list_status":
            return cls.get_list_status_from_entry(entry)
        elif stat_name == "num_watched":
            return cls.get_num_watched_from_entry(entry)
        elif stat_name == "id":
            return cls.get_id_from_entry(entry)
        elif stat_name == "updated_at":
            return cls.get_updated_at_from_entry(entry)


class MALEntryHandler(AnimeEntryHandler):
    def __init__(self):
        pass

    @staticmethod
    def get_title_from_entry(entry):
        try:
            return entry['node']['title']
        except TypeError:
            pass

    @staticmethod
    def get_score_from_entry(entry):
        return entry['list_status']['score']

    @staticmethod
    def get_list_status_from_entry(entry):
        return entry['list_status']['status']

    @staticmethod
    def get_num_watched_from_entry(entry):
        return entry['list_status']['num_episodes_watched']

    @staticmethod
    def get_id_from_entry(entry):
        return entry['node']['id']

    @staticmethod
    def get_updated_at_from_entry(entry):
        return entry['list_status']['updated_at']


class AnilistEntryHandler(AnimeEntryHandler):
    def __init__(self):
        pass

    @staticmethod
    def get_title_from_entry(entry):
        entry_mal_id = entry['media']['idMal']
        mal_title = MALUtils.get_anime_title_by_id(entry_mal_id)
        return mal_title

    @staticmethod
    def get_score_from_entry(entry):
        return entry['score']

    @staticmethod
    def get_list_status_from_entry(entry):
        return MALUtils.convert_anilist_status_to_MAL(entry['status'])

    @staticmethod
    def get_num_watched_from_entry(entry):
        return entry['media']['episodes']
        # Anilist currently doesn't have num watched info, this returns
        # episode number

    @staticmethod
    def get_id_from_entry(entry):
        return entry['media']['idMal']

    @staticmethod
    def get_updated_at_from_entry(entry):
        unix_timestamp = entry['updatedAt']
        datetime_obj = datetime.utcfromtimestamp(unix_timestamp)
        updated_at = datetime_obj.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        # Anilist returns a Unix timestamp, synchronized this with the MAL timestamp format
        return updated_at
