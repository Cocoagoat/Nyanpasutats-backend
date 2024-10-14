import os

from colorama import Fore
from main.modules.AnimeDB import AnimeDB
from main.modules.general_utils import save_pickled_file, load_pickled_file, get_data
from main.modules.MAL_utils import MALUtils
from main.modules.GlobalValues import MINIMUM_SCORE
from igraph import Graph
import networkx as nx
import matplotlib.pyplot as plt
from main.modules.filenames import *
from main.modules.Errors import UserDoesNotExistError
import copy
import logging
import requests
import time

logger = logging.getLogger("nyanpasutats")


class AnimeGraph(Graph):

    def __init__(self, *args, **kwargs):
        self._titles = None
        super().__init__(*args, **kwargs)

    def __eq__(self, G2):
        if self.vs['name'] == G2.vs['name'] and len(self.es) == len(G2.es):
            return True
        return False

    @property
    def titles(self):
        return [] if not self.vs else self.vs['name']

    def delete_edge_if_exists(self, index1, index2):
        edge_to_delete = self.get_eid(index1, index2, directed=True, error=False)
        if edge_to_delete != -1:
            self.delete_edges(edge_to_delete)

    def inplace_union(self, G2):  # Make this not create duplicate edges
        for vertex in G2.vs:
            if vertex['name'] not in self.titles:
                self.add_vertex(name=vertex['name'])

        for vertex in G2.vs:
            for adjacent_vertex in vertex.neighbors():
                if G2.are_connected(vertex, adjacent_vertex):
                    edge_id = G2.get_eid(vertex, adjacent_vertex)
                    edge = G2.es[edge_id]
                    self.add_edge(vertex['name'], adjacent_vertex['name'], relation=edge['relation'])

    def split(self, update=False):
        """This function splits a graph of a group of anime related to each other in some way on MAL.
        The split is done by whether the related shows are actually the same show, or are related for
        some other reason (shared characters or such). For more details see description of
        are_separate_shows"""
        try:
            anime_db = AnimeDB(anime_database_updated_name)
        except FileNotFoundError:
            anime_db = AnimeDB()

        G = self.copy()
        edges_to_delete = []
        for vertex in G.vs:
            vertex_edge_list = vertex.incident()
            for edge in vertex_edge_list:
                try:
                    v1, v2 = edge.vertex_tuple
                except ValueError:
                    continue

                relation_type = edge['relation']
                if anime_db.are_separate_shows(v1['name'], v2['name'], relation_type
                                               ) or self.separate_show_exists_in_subgraph(v1, v2,
                                                                                     relation_type):
                    edges_to_delete.append((v1.index, v2.index))
                    edges_to_delete.append((v2.index, v1.index))

        for v1, v2 in edges_to_delete:
            G.delete_edge_if_exists(v1, v2)

        if len(G.es) != len(self.es):
            return G.split()
        return G

    def separate_show_exists_in_subgraph(self, v1, v2, relation_type):
        try:
            anime_db = AnimeDB(anime_database_updated_name)
        except FileNotFoundError:
            anime_db = AnimeDB()

        if relation_type not in ['other', 'spin-off', 'side_story', 'parent_story']:
            return False

        show_lengths = anime_db.get_stats_of_shows([v1['name'], v2['name']], ['Episodes', 'Duration'])
        try:
            v1_length = show_lengths[v1['name']]['Episodes'] * show_lengths[v1['name']]['Duration']
            v2_length = show_lengths[v2['name']]['Episodes'] * show_lengths[v2['name']]['Duration']
        except KeyError:
            return False

        longer_show, shorter_show = (v1['name'], v2['name']) if v1_length > v2_length else (v2['name'], v1['name'])

        G_copy = copy.deepcopy(self)

        old_subgraphs = G_copy.connected_components(mode='WEAK').subgraphs()

        G_copy.delete_edge_if_exists(v1, v2)
        G_copy.delete_edge_if_exists(v2, v1)

        connected_components = G_copy.connected_components(mode='WEAK')
        if len(connected_components) == len(old_subgraphs):
            return False

        new_subgraphs = connected_components.subgraphs()
        for new_graph in new_subgraphs:
            if shorter_show in new_graph.vs['name']:
                subgraph = new_graph

        for v in subgraph.vs:
            if anime_db.are_separate_shows(longer_show, v['name'], relation_type):
                return True

        return False

    def delete_duplicate_edges(self):
        for edge1 in self.es:
            for edge2 in self.es:
                if edge1 != edge2 and edge1.target == edge2.target and edge1.source == edge2.source:
                    self.delete_edges(edge2.index)

    def determine_main_show(self, update=False):
        """The main show for each graph will be the one with the most members (to avoid graphs
        having an "Attack on Titan : Snickers Collab" key)"""
        anime_db = AnimeDB(#None if not update else
             anime_database_updated_name)
        members_of_each_show = anime_db.get_stats_of_shows(self.titles, ['Scores'])
        # Return the entry which the maximum amount of people watched out of all related entries
        try:
            return max(members_of_each_show, key=lambda x: members_of_each_show[x]['Scores'])
        except ValueError:
            logging.error(f"Unable to determine main show for graph with titles {self.titles}. Substituting.")
            return self.titles[0]  # Not something critical enough to make the program stop running,
            # but warrants a manual look eventually

    def get_adjacent_vertex_names(self, vertex_name: str):
        """ This function returns the names of the vertices adjacent to a known vertex in a graph G."""
        adj_vertex_list = []
        incident_edges = self.incident(vertex_name)  # Incident = all the edges V is in
        for edge_index in incident_edges:
            edge = self.es[edge_index]
            source_vertex_index, target_vertex_index = edge.tuple
            adj_vertex = self.vs[target_vertex_index]['name']
            adj_vertex_list.append(adj_vertex)
        return adj_vertex_list

    def plot_graph(self):
        """This essentially plots the graph in a way that the relation types between each two vertices
         can be clearly seen as long as the graph isn't too crowded. Last part is courtesy of ChatGPT,
         I honestly have no idea how it thought of this very specific label_pos."""

        labels_list = self.titles
        node_labels = {i: label for i, label in enumerate(labels_list)}

        # Converting the graph to a different library since igraph x matplotlib caused some issues.
        networkx_graph = self.to_networkx()

        # Plotting the graph itself in a separate 5120x2880 window (biggest possible size to see as much detail
        # as possible, even this won't be enough for the most crowded graphs).
        pos = nx.spring_layout(networkx_graph)
        plt.figure(figsize=(51.2, 28.8))
        nx.draw_networkx_nodes(networkx_graph, pos, node_size=200, node_color='skyblue')
        nx.draw_networkx_labels(networkx_graph, pos, labels=node_labels, font_size=6)
        nx.draw_networkx_edges(networkx_graph, pos, edgelist=networkx_graph.edges, edge_color='gray',
                               connectionstyle=f"arc{3},rad={0.1}")

        # This part plots the labels on the edges
        for edge, label in nx.get_edge_attributes(networkx_graph, 'relation').items():
            start, end = pos[edge[0]], pos[edge[1]]
            mid = (start + end) / 2
            label_pos = mid + 0.1 * (end - start)[::-1] * [-1, 1] * (hash(edge) % 2 * 2 - 1)
            plt.text(label_pos[0], label_pos[1], label, horizontalalignment='center', verticalalignment='center',
                     fontsize=4, weight='bold', color='blue')
        plt.show()


class GraphCollection():

    def __init__(self, graphs=None):
        self.graphs = graphs if graphs else {}
        self._related_shows = None
        self._all_titles = None

    def keys(self):
        return self.graphs.keys()

    def values(self):
        return self.graphs.values()

    def items(self):
        return self.graphs.items()

    def pop(self, x):
        return self.graphs.pop(x)

    @property
    def related_shows(self):
        if not self._related_shows or len(self._related_shows) != len(self.graphs):
            self._related_shows = {}
            for key, graph in self.graphs.items():
                self._related_shows[key] = graph.vs['name']
        return self._related_shows

    @property
    def all_titles(self):
        if not self._all_titles:
            self._all_titles = [title for graph in self.graphs.values() for title in graph.titles]
        return self._all_titles

    @all_titles.setter
    def all_titles(self, value):
        self._all_titles = value

    def find_graph_by_title(self, title):  # Put this outside?
        for key, graph in self.graphs.items():
            if title in graph.titles:
                return key, graph

    def split_graphs(self, affected_keys=None):
        new_graph_dict = {}
        count = 0
        graph_dict_size = len(self.graphs)
        for key, graph in self.graphs.items():
            count = count + 1
            if affected_keys is not None and key not in affected_keys:
                # In case of an update, we don't want re-split all 238794234 graphs,
                # only those affected by the update
                new_graph_dict[key] = graph
                continue
            print(f"Currently on graph {count} out of {graph_dict_size}")
            new_split_graph = graph.split()
            connected_components = new_split_graph.connected_components(mode='WEAK')
            new_graphs = connected_components.subgraphs()

            for graph in new_graphs:
                try:
                    main_show = graph.determine_main_show(update=True if affected_keys else False)
                except BaseException:  # temporary
                    print(f"Error determining main show in graph, the affected shows are : {graph.titles}")
                    logger.info(f"Error determining main show in graph, the affected shows are : {graph.titles}")
                    main_show = graph.titles[0]
                new_graph_dict[main_show] = graph
        new_graph_dict = sorted(new_graph_dict.items(), key=lambda x: len(x[1].titles),
                                reverse=True)

        new_graph_dict = {x[0]: x[1] for x in new_graph_dict}
        self.graphs = new_graph_dict

    def find_related_entries(self, show_name):
        if show_name in self.related_shows.keys():
            return show_name, self.related_shows[show_name]
        else:
            # It's more efficient to search through the keys first,
            # since keys are the root entry for each show, and most shows
            # only have one major entry (one season/movie).
            for root_name in self.related_shows.keys():
                if show_name in self.related_shows[root_name]:
                    return root_name, self.related_shows[root_name]

    def filter_low_scores(self):
        anime_db = AnimeDB(anime_database_updated_name)
        #  can use the updated in both cases here, even if we get a show that isn't supposed to be in
        #  a non-updated GraphCollection it'll get filtered out by [x for x in self.all_titles]
        relevant_titles = anime_db.filter_titles(AnimeDB.show_meets_standard_conditions_nls)
        irrelevant_titles = [x for x in self.all_titles
                             if x not in relevant_titles]

        self.remove_entries(irrelevant_titles)

    def remove_entries(self, entries_to_remove, update=False):
        old_keys = self.graphs.keys()
        new_keys = []
        self._all_titles = None
        for key, graph in self.graphs.items():
            # For each graph, remove the vertices that don't
            # meet the score condition
            for title in graph.titles:
                if title in entries_to_remove:
                    graph.delete_vertices(title)

            if key not in graph.titles:
                # If a main show (a key) was deleted and its graph still has some shows left,
                # (for example, this could happen if S1 of a show had a score of 6.4 but
                # S2 has a score of 6.6), we need to determine the new main show.
                if graph:
                    main_show = graph.determine_main_show(update=update)
                    new_keys.append(main_show)
                # If a show was deleted and its graph is now empty, we only need to put an
                # empty space in new_keys so that the order of the graphs get preserved when
                # the new dictionary gets created from new_keys. We will then delete this
                # dummy key.
                else:
                    new_keys.append("")
            else:
                # If the main show wasn't deleted, we'll simply re-attach the new graph to it.
                new_keys.append(key)

        self.graphs = {new_key: self.graphs.pop(old_key)
                       for old_key, new_key in list(zip(old_keys, new_keys))}
        self.graphs.pop('', 0)


class GraphCreator():

    def __init__(self, titles, update=False, update_from_scratch=False, titles_to_remove=[]):
        try:
            if not update:
                self._graph_collection = load_pickled_file(temp_graphs_dict_filename)
            else:
                if os.path.exists(graphs_dict_updated_filename) and not update_from_scratch:
                    # print("Went into the correct if, initial graphs are the updated ones")
                    self._graph_collection = load_pickled_file(graphs_dict_updated_filename)
                else:
                    self._graph_collection = load_pickled_file(graphs_dict_filename)

                if titles_to_remove:
                    self._graph_collection.remove_entries(titles_to_remove, update=update)

            self.processed_anime = list(set([x for G in list(self._graph_collection.graphs.values())
                                             for x in G.vs['name']]))
        except FileNotFoundError:
            if update:
                raise FileNotFoundError("Original graphs file not found")
            self._graph_collection = GraphCollection()
            self.processed_anime = []

        self.anime_db = AnimeDB(filename=None if not update else anime_database_updated_name)
        # print(f"Titles in GraphCreator : {titles}")
        # print(f"Length of updated df inside GraphCreator : {len(self.anime_db.df.columns)}")
        self.relevant_titles = self.anime_db.sort_titles_by_release_date(titles)
        # print(f"Relevant titles in GraphCreator : {self.relevant_titles}")

        self.update = update
        self.saved_for_update = False

    def traverse_anime_relations(self, anime_id: int, G: AnimeGraph):

        def save_graphs_for_update():
            """Saves the graphs object of everything before the current season.
            This object will be used as a starting point to update the actual
            graphs object every day in production so that new non-sequels are
            properly added to shows_tags_dict daily as well."""
            current_season = MALUtils.get_current_season()

            anime_year = current_anime['start_season'].get('year', 0)
            anime_season = current_anime['start_season'].get('season', '')

            if current_season['Year'] == anime_year and current_season['Season'].name.lower() == anime_season.lower():
                self.saved_for_update = True
                save_pickled_file(data_path / "graphs_for_daily_update.pickle", self._graph_collection)

        url = f'https://api.myanimelist.net/v2/anime/{int(anime_id)}?fields=id,media_type,mean,' \
              f'related_anime,genres,' \
              f'average_episode_duration,num_episodes,num_scoring_users,start_season'

        current_anime = None
        while not current_anime:
            try:
                print(f"Getting data for anime ID {anime_id}")
                current_anime = get_data(url)
                time.sleep(1)  # extra sleep just in case
            except (TimeoutError, requests.exceptions.ReadTimeout) as e:
                current_anime = None
            except UserDoesNotExistError:
                return

        title = current_anime['title']
        print(Fore.LIGHTWHITE_EX + f"Currently on title {title}")
        G.add_vertex(name=title)
        self.processed_anime.append(title)

        # Until here we can't know whether the show is related to any previous ones

        for related_anime in current_anime['related_anime']:
            related_title = related_anime['node']['title']
            if related_title not in self.relevant_titles and (related_title not in
                                                              self.processed_anime or not self.update):
                continue
            relation_type = related_anime['relation_type']

            if related_title not in self.processed_anime:
                self.traverse_anime_relations(related_anime['node']['id'], G)

            if related_title in G.vs['name']:
                # related_title was already added to G by a previous recursion of traverse,
                # this is the normal case.
                G.add_edge(title, related_title, relation=relation_type)
            else:
                # The much more complex case. This means that related_title is already part of another graph,
                # which title for some reason isn't (despite being related to related_title). This will most often
                # happen when updating an existing graph - for example, a new entry to the Crayon Shin-chan franchise
                # got added to MAL in the past week, but all the other 23904230423 entries already exist in another graph.
                # Hence we should add this new entry to the existing graph, and not create a new one.

                root, G_existing = self._graph_collection.find_graph_by_title(related_title)

                # If "delet dis" is in G, that means G was marked for deletion during another recursion
                # because it was already merged with another existing graph. This "deletion" will be done
                # simply by replacing graph_collection[old root] = old G with graph_collection[root] = G,
                # with G now containing old G inside it (but likely having a different root, since the root of all
                # vertices in G is the anime with the most members in it, and that's likely to be in the old
                # graph

                # In other words this will happen if there are several new entries to a franchise that already
                # exists in another graph
                if "delet dis" in G.vs['name']:
                    root = G.get_adjacent_vertex_names("delet dis")[0]
                    G.delete_vertices("delet dis")

                # We add the new vertex to the existing graph
                G_existing.add_vertex(title)
                G_existing.add_edge(title, related_title, relation=relation_type)

                G.inplace_union(G_existing)
                self._graph_collection.graphs[root] = G.copy()

                # The new graph is marked for deletion, since it got merged with the existing one.
                # It will be deleted in create_graphs().
                G.add_vertex(name="delet dis")
                if root not in G.vs['name']:
                    G.add_vertex(name=root)
                G.add_edge("delet dis", root)

                print(f"Deleting graph. Graph's vertices are {G.vs['name']}")

            if len(self._graph_collection.graphs) - 1 % 20 == 0 or title == self.relevant_titles[-1]:
                save_pickled_file(temp_graphs_dict_filename, self._graph_collection)

    def create_graphs(self, filename=None):

        relevant_ids = [self.anime_db.get_id_by_title(title) for title in self.relevant_titles]
        # print(f"Relevant titles in create_graphs : {self.relevant_titles}")
        # print(f"Relevant ids in create_graphs : {relevant_ids}")
        for title, ID in list(zip(self.relevant_titles, relevant_ids)):
            if title in self.processed_anime:  # or MAL_title not in relevant_titles:
                continue

            G = AnimeGraph(directed=True)
            self.traverse_anime_relations(ID, G)
            if len(G.vs) == 0:
                continue

            if 'delet dis' not in G.vs['name']:
                self._graph_collection.graphs[title] = G
            else:
                # 'delet dis' was attached only to the root in traverse_anime_relations
                root = G.determine_main_show(update=self.update)
                G.delete_vertices("delet dis")
                self._graph_collection.graphs[root] = G

                # G in this case will be comprised of one or more old graphs + new additions.
                # All the old ones need to be deleted since they were consolidated into the
                # new one.
                keys_to_delete = [key for key in self._graph_collection.graphs.keys()
                                  if key in G.titles and key != root]
                [self._graph_collection.pop(key) for key in keys_to_delete]

            if len(self._graph_collection.graphs) % 5 == 0 or title == self.relevant_titles[-1]:
                save_pickled_file(temp_graphs_dict_filename, self._graph_collection)

        for _, graph in self._graph_collection.items():
            graph.delete_duplicate_edges()

        if self.update:
            unsplit_graphs_filename = "unsplit_graphs-U.pickle"
            filename = graphs_dict_updated_filename
            # Recreate all_titles since we may have added new graphs
            self._graph_collection.all_titles = [title for graph in self._graph_collection.values() for title in graph.titles]
        else:
            unsplit_graphs_filename = "unsplit_graphs.pickle"
            filename = graphs_dict_filename

        save_pickled_file(data_path / unsplit_graphs_filename, self._graph_collection)
        self._graph_collection.split_graphs(affected_keys=self.relevant_titles)

        save_pickled_file(filename, self._graph_collection)
        return self._graph_collection


class Graphs:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._all_graphs = None
            cls._instance._all_graphs_nls = None
            cls._instance._all_graphs_updated = None
            cls._instance._all_graphs_nls_updated = None
            cls._instance.anime_db = AnimeDB()

        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def initialize_graph_collection(self, no_low_scores=False):
        filename = graphs_dict_nls_filename if no_low_scores else graphs_dict_filename
        try:
            print("Loading anime graphs")
            if no_low_scores:
                self._all_graphs_nls = load_pickled_file(filename)
            else:
                self._all_graphs = load_pickled_file(filename)
            print("Anime graphs dictionary loaded successfully")

        except FileNotFoundError:
            print("Anime graphs dictionary not found. Creating new anime graphs dictionary")

            def condition_func(show_stats: dict):
                if (int(show_stats[AnimeDB.stats["Scores"]]) >= 2000
                    or AnimeDB.show_is_from_current_season(show_stats)) \
                        and show_stats[AnimeDB.stats["Duration"]] * \
                        show_stats[AnimeDB.stats["Episodes"]] >= 15 \
                        and show_stats[AnimeDB.stats["Duration"]] >= 3 \
                        and (not no_low_scores or show_stats[AnimeDB.stats["Mean Score"]] >= MINIMUM_SCORE):
                    return True
                return False

            relevant_titles = self.anime_db.filter_titles(condition_func)
            if no_low_scores:
                all_graphs_copy = copy.deepcopy(self.all_graphs)
                all_graphs_copy.filter_low_scores()
                self._all_graphs_nls = all_graphs_copy
                save_pickled_file(graphs_dict_nls_filename, self._all_graphs_nls)
            else:
                graph_creator = GraphCreator(relevant_titles)
                self._all_graphs = graph_creator.create_graphs()

    @staticmethod
    def update_graphs(self, titles_to_add, titles_to_remove, update_from_scratch=False):
        graph_creator = GraphCreator(titles_to_add, update=True,
                                     update_from_scratch=update_from_scratch,
                                     titles_to_remove=titles_to_remove)
        graph_creator.create_graphs()

    @property
    def all_graphs(self):
        if not self._all_graphs:
            self.initialize_graph_collection(no_low_scores=False)
        return self._all_graphs

    @property
    def all_graphs_nls(self):
        if not self._all_graphs_nls:
            self.initialize_graph_collection(no_low_scores=True)
        return self._all_graphs_nls

    @property
    def all_graphs_updated(self):
        if not self._all_graphs_updated:
            try:
                self._all_graphs_updated = load_pickled_file(graphs_dict_updated_filename)
            except FileNotFoundError:
                try:
                    self.update_graphs()
                except FileNotFoundError:
                    self.initialize_graph_collection(no_low_scores=False)
        return self._all_graphs_updated

    @property
    def all_graphs_nls_updated(self):
        if not self._all_graphs_nls_updated:
            try:
                self._all_graphs_nls_updated = load_pickled_file(graphs_dict_nls_updated_filename)
            except FileNotFoundError:
                self._all_graphs_nls_updated = copy.deepcopy(self.all_graphs_updated)
                self._all_graphs_nls_updated.filter_low_scores()
                save_pickled_file(graphs_dict_nls_updated_filename, self._all_graphs_nls_updated)
        return self._all_graphs_nls_updated






