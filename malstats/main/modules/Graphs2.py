from colorama import Fore
from main.modules.AnimeDB import AnimeDB
from main.modules.general_utils import *
from igraph import Graph
import networkx as nx
import matplotlib.pyplot as plt
from main.modules.filenames import *
import os
import copy


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

    def inplace_union(self, G2):  # Put this outside?
        for vertex in G2.vs:
            if vertex['name'] not in self.titles:
                self.add_vertex(name=vertex['name'])

        for vertex in G2.vs:
            for adjacent_vertex in vertex.neighbors():
                if G2.are_connected(vertex, adjacent_vertex):
                    edge_id = G2.get_eid(vertex, adjacent_vertex)
                    edge = G2.es[edge_id]
                    self.add_edge(vertex['name'], adjacent_vertex['name'], relation=edge['relation'])

    def split(self):
        """This function splits a graph of a group of anime related to each other in some way on MAL.
        The split is done by whether the related shows are actually the same show, or are related for
        some other reason (shared characters or such). For more details see description of
        are_separate_shows"""
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

    def determine_main_show(self):
        """The main show for each graph will be the one with the most members (to avoid graphs
        having an "Attack on Titan : Snickers Collab" key)"""
        anime_db = AnimeDB()
        members_of_each_show = anime_db.get_stats_of_shows(self.titles, ['Scores'])
        # Return the entry which the maximum amount of people watched out of all related entries
        return max(members_of_each_show, key=lambda x: members_of_each_show[x]['Scores'])

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
        # as possible, even this won't be enough for the most crowded graphs.
        pos = nx.spring_layout(networkx_graph)
        plt.figure(figsize=(51.2, 28.8))
        nx.draw_networkx_nodes(networkx_graph, pos, node_size=200, node_color='skyblue')
        nx.draw_networkx_labels(networkx_graph, pos, labels=node_labels, font_size=6)
        nx.draw_networkx_edges(networkx_graph, pos, edgelist=networkx_graph.edges, edge_color='gray',
                               connectionstyle=f"arc{3},rad={0.1}")

        # This part plots the labels on the edges, courtesy of ChatGPT.
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

    def find_graph_by_title(self, title):  # Put this outside?
        for key, graph in self.graphs.items():
            if title in graph.titles:
                return key, graph

    def split_graphs(self):
        new_graph_dict = {}
        count = 1
        graph_dict_size = len(self.graphs)
        for key, graph in self.graphs.items():
            print(f"Currently on graph {count} out of {graph_dict_size}")
            new_split_graph = graph.split()
            connected_components = new_split_graph.connected_components(mode='WEAK')
            new_graphs = connected_components.subgraphs()
            count = count + 1

            for graph in new_graphs:
                try:
                    main_show = graph.determine_main_show()
                except BaseException:  # temporary
                    print(f"ERROR, the affected shows are : {graph.titles}")
                new_graph_dict[main_show] = graph

        new_graph_dict = sorted(new_graph_dict.items(), key=lambda x: len(x[1].titles),
                                reverse=True)

        new_graph_dict = {x[0]: x[1] for x in new_graph_dict}
        self.graphs = new_graph_dict

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

    def remove_entries(self, entries_to_remove):
        old_keys = self.graphs.keys()
        new_keys = []
        self._all_titles = None
        for key, graph in self.graphs.items():
            # For each graph, remove the vertices that don't meet the score condition
            for title in graph.titles:
                if title in entries_to_remove:
                    graph.delete_vertices(title)

            if key not in graph.vs['name']:
                # If a main show (a key) was deleted and its graph still has some shows left,
                # (for example, this could happen if S1 of a show had a score of 6.4 but
                # S2 has a score of 6.6), we need to determine the new main show.
                if graph:
                    main_show = graph.determine_main_show()
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
        self.graphs.pop('')
        save_pickled_file(graphs_dict_no_low_scores_filename, self)


class GraphCreator():

    def __init__(self, titles):
        try:
            self._graph_collection = load_pickled_file(temp_graphs_dict_filename)
            self.processed_anime = list(set([x for G in list(self._graph_collection.graphs.values())
                                             for x in G.vs['name']]))
        except FileNotFoundError:
            self._graph_collection = GraphCollection()
            self.processed_anime = []

        self.anime_db = AnimeDB()
        self.relevant_titles = self.anime_db.sort_titles_by_release_date(titles)

    def traverse_anime_relations(self, anime_id: int, G: AnimeGraph):

        url = f'https://api.myanimelist.net/v2/anime/{int(anime_id)}?fields=id,media_type,mean,' \
              f'related_anime,genres,' \
              f'average_episode_duration,num_episodes,num_scoring_users'

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
            if related_title not in self.relevant_titles:
                continue
            relation_type = related_anime['relation_type']

            if related_title not in self.processed_anime:
                self.traverse_anime_relations(related_anime['node']['id'], G)

            if related_title in G.vs['name']:  # If traverse was activated before (first time processing the anime)
                # then the vertex should definitely be in the graph. Otherwise, this isn't the first time.
                G.add_edge(title, related_title, relation=relation_type)
            else:
                root, G_existing = self._graph_collection.find_graph_by_title(related_title)
                if "delet dis" in G.vs['name']:
                    root = G.get_adjacent_vertex_names("delet dis")[0]
                    G.delete_vertices("delet dis")

                G_existing.add_vertex(title)
                G_existing.add_edge(title, related_title, relation=relation_type)

                G.inplace_union(G_existing)
                self._graph_collection.graphs[root] = G.copy()

                # maybe problem is here - check ID of G and graph_dict[root] after the eq, maybe use copy()
                G.add_vertex(name="delet dis")
                if root not in G.vs['name']:
                    G.add_vertex(name=root)
                G.add_edge("delet dis", root)

                print(f"Deleting graph. Graph's vertices are {G.vs['name']}")

            if len(self._graph_collection.graphs) - 1 % 20 == 0 or title == self.relevant_titles[-1]:
                save_pickled_file(temp_graphs_dict_filename, self._graph_collection)

    def create_graphs(self, filename=None):

        relevant_ids = [self.anime_db.get_id_by_title(title) for title in self.relevant_titles]

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
                # This handles a very specific case, need to elaborate later
                root = G.get_adjacent_vertex_names("delet dis")[0]
                G.delete_vertices("delet dis")
                self._graph_collection.graphs[root] = G

            if len(self._graph_collection.graphs) % 5 == 0 or title == self.relevant_titles[-1]:
                save_pickled_file(temp_graphs_dict_filename, self._graph_collection)

        for _, graph in self._graph_collection.items():
            graph.delete_duplicate_edges()

        save_pickled_file(data_path / "unsplit_graphs.pickle", self._graph_collection)
        self._graph_collection.split_graphs()

        filename = filename if filename else graphs_dict_filename
        save_pickled_file(filename, self._graph_collection)

        os.remove(temp_graphs_dict_filename)
        return self._graph_collection


class Graphs2:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._all_graphs = None
            cls._instance._all_graphs_no_low_scores = None
            # cls._instance.anime_db = AnimeDB()

        return cls._instance

    def __init__(self):
        self.anime_db = AnimeDB()

    def initialize_graph_collection(self, no_low_scores=False):
        filename = graphs_dict_no_low_scores_filename if no_low_scores else graphs_dict_filename
        try:
            print("Loading anime graphs")
            if no_low_scores:
                self._all_graphs_no_low_scores = load_pickled_file(filename)
            else:
                self._all_graphs = load_pickled_file(filename)
            print("Anime graphs dictionary loaded successfully")

        except FileNotFoundError:
            print("Anime graphs dictionary not found. Creating new anime graphs dictionary")

            def condition_func(show_stats: dict):
                if int(show_stats[AnimeDB.stats["Scores"]]) >= 2000 \
                        and show_stats[AnimeDB.stats["Duration"]] * \
                        show_stats[AnimeDB.stats["Episodes"]] >= 15 \
                        and show_stats[AnimeDB.stats["Duration"]] >= 3 \
                        and (not no_low_scores or show_stats[AnimeDB.stats["Mean Score"]] >= 6.5):
                    # and show_stats[AnimeDB.stats["Year"]] >= 2016 \
                    #                  and show_stats[AnimeDB.stats["Year"]] <= 2017:

                    # and show_stats[AnimeDB.stats["Scores"]]<=80000:
                    return True
                return False

            relevant_titles = self.anime_db.filter_titles(condition_func)
            if no_low_scores:
                irrelevant_titles = [x for x in self.all_graphs.all_titles
                                     if x not in relevant_titles]
                self._all_graphs_no_low_scores = copy.deepcopy(self.all_graphs)
                self._all_graphs_no_low_scores.remove_entries(irrelevant_titles)

            else:
                graph_creator = GraphCreator(relevant_titles)
                self._all_graphs = graph_creator.create_graphs(filename=filename)

    @property
    def all_graphs(self):
        if not self._all_graphs:
            self.initialize_graph_collection(no_low_scores=False)
        return self._all_graphs

    @property
    def all_graphs_no_low_scores(self):
        if not self._all_graphs_no_low_scores:
            self.initialize_graph_collection(no_low_scores=True)
        return self._all_graphs_no_low_scores


