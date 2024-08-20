from main.modules.AnimeEntryHandler import AnimeEntryHandler


class ListFormatter():
    def __init__(self, anime_list, stats_to_get=None):
        # anime_list should be of type AnimeList
        site = AnimeEntryHandler.determine_list_site(anime_list)
        self.anime_entry_handler = AnimeEntryHandler.get_concrete_handler(site)
        self.anime_list = anime_list
        self._formatted_list = {}
        all_stats = ["score", "list_status", "num_watched", "id", "updated_at"]
        self.stats_to_get = stats_to_get if stats_to_get else all_stats

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



