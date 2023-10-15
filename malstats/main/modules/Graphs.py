from .UserDB import UserDB
from .AnimeDB import AnimeDB
from .general_utils import *
from igraph import Graph, summary, union
import networkx as nx
import matplotlib.pyplot as plt
from .filenames import *


class Graphs:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """The class is a Singleton - we only need one instance of it since its purpose is
        to house and create on demand all the data structures that are used in this project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # All properties are loaded on demand
        self._all_graphs = None
        self._related_shows = None
        self.anime_db = AnimeDB()

    @property
    def all_graphs(self):
        if not isinstance(self._all_graphs, dict):
            try:
                print("Loading anime graphs")
                self._all_graphs = load_pickled_file(graphs_dict_filename)
                print("Anime graphs dictionary loaded successfully")
            except FileNotFoundError:
                print("Anime graphs dictionary not found. Creating new anime graphs dictionary")
                self.create_graph_of_all_anime()
        return self._all_graphs

    @property
    def related_shows(self):
        if not isinstance(self._related_shows, dict):
            self._related_shows = {}
            for key, graph in self.all_graphs.items():
                self._related_shows[key] = graph.vs['name']
        return self._related_shows

    @staticmethod
    def plot_graph(G):
        """This essentially plots the graph in a way that the relation types between each two vertices
         can be clearly seen as long as the graph isn't too crowded. Last part is courtesy of ChatGPT,
         I honestly have no idea how it thought of this very specific label_pos."""

        labels_list = G.vs['name']
        node_labels = {i: label for i, label in enumerate(labels_list)}

        # Converting the graph to a different library since igraph x matplotlib caused some issues.
        networkx_graph = G.to_networkx()
        edge_labels = nx.get_edge_attributes(networkx_graph, 'relation')

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

    @staticmethod
    def get_adjacent_vertex_names(G, vertex_name: str):
        """ This function returns the names of the vertices adjacent to a known vertex in a graph G."""
        adj_vertex_list = []
        incident_edges = G.incident(vertex_name)  # Incident = all the edges V is in
        for edge_index in incident_edges:
            edge = G.es[edge_index]
            source_vertex_index, target_vertex_index = edge.tuple
            adj_vertex = G.vs[target_vertex_index]['name']
            adj_vertex_list.append(adj_vertex)
        return adj_vertex_list

    def are_separate_shows(self, show1: str, show2: str, relation_type: str):
        """ This method tries to determine whether two entries that are related in some way on MAL
        are the same show, or two different shows.

        Note : The methodology used here is very rough, and relies purely on how the shows are related
        to each other, their media type, and their length. There are two reasons for this :

        1) Even the definition of same show vs different shows is very rough - for example,
        is Fate/Zero the same show as Fate/Unlimited Blade Works? Is A Certain Magical Index
        the same show as A Certain Scientific Railgun?

        2) Even if you count Fate/Zero and Fate/UBW as different shows, there is literally no way
        to separate them going purely by their MAL entries. Fate/UBW is listed as a sequel of Fate/Zero,
        both are full-length TV shows, and both have a very sizeable watcher amount. There are multiple cases
        of shows like these where it's simply impossible to separate them due to how MAL classifies them
        (sometimes outright misclassifies, like putting alternative_version (which should NOT count as a
        separate show, since alternative version is basically the same show but made in a different time/
        from a different perspective) instead of alternative_setting (which usually means same universe but
        completely different characters, and would almost always be a different show).

        In short, we can only rely on non-fully-accurate MAL data to separate what would be difficult
        even for humans to agree on, so this won't be 100% precise.

         """

        def both_shows_are_TV():
            return show_stats[show1]["Type"] == 1 and show_stats[show2]["Type"] == 1

        def both_shows_are_movies():
            return show_stats[show1]["Type"] == 2 and show_stats[show2]["Type"] == 2

        def show_is_longer_than(minutes, name):
            if not show_stats[name]["Episodes"]:
                show_stats[name]["Episodes"] = 1
            # if show_stats[name]["Duration"]==1:
            #     show_stats[name]["Duration"]=65
            return show_stats[name]["Duration"] * show_stats[name]["Episodes"] > minutes

        if show1 not in self.anime_db.titles or show2 not in self.anime_db.titles:  # take care of this outside later
            return True

        relevant_stats = ["Duration", "Episodes", "Type"]
        show_stats = self.anime_db.get_stats_of_shows([show1, show2], relevant_stats)
        # Put these into the 3rd case^
        if relation_type in ['sequel', 'prequel', 'alternative_version', 'summary']:
            # Sequels, prequels, alternative versions and summaries are never separate shows
            return False

        if relation_type == 'character':
            # "character" means that the only common thing between the two shows is that some of
            # the characters are mutual. It will always be a separate show, or something very short
            # that isn't in the partial database in the first place.
            return True

        if relation_type in  ['other' | 'side_story' | 'spin_off']:  # add parent?
            # This is the most problematic case. MAL is very inconsistent with how it labels things
            # as "other", "side story" or "spin-off". The latter two are used almost interchangeably,
            # and "other" can be used for pretty much ANYTHING. Side stories/spin-offs, commercials,
            # even crossovers. There is no feasible way to catch literally every case, but this gets
            # the vast majority of them.
            if both_shows_are_TV() or both_shows_are_movies() or \
                    (show_is_longer_than(144, show1) and show_is_longer_than(144, show2)):
                return True  # Add a search for what sequels are?
            return False

        if relation_type == 'alternative_setting':
                # Alternative setting almost always means that the shows are set in the same universe,
                # but have different stories or even characters. Sometimes it can also be used for
                # miscellanous related shorts, which is why I made a small (arbitrary) length requirement
                # for the shows to be counted as separate.

            if (show_is_longer_than(60, show1) and show_is_longer_than(60, show2)):
                return True
            return False

        return False

    def split_graph(self, G):
        """This function splits a graph of a group of anime related to each other in some way on MAL.
        The split is done by whether the related shows are actually the same show, or are related for
        some other reason (shared characters or such). For more details see description of
        are_separate_shows"""
        G = G.copy()
        edges_to_delete = []
        for vertex in G.vs:
            vertex_edge_list = vertex.incident()
            for edge in vertex_edge_list:
                v1, v2 = edge.vertex_tuple

                relation_type = edge['relation']
                if self.are_separate_shows(v1['name'], v2['name'], relation_type):
                    edges_to_delete.append((v1.index, v2.index))
                    edges_to_delete.append((v2.index, v1.index))

        for v1, v2 in edges_to_delete:
            edge_to_delete = G.get_eid(v1, v2, directed=True, error=False)
            if edge_to_delete != -1:
                G.delete_edges(edge_to_delete)
        return G

    def split_graphs(self, graph_dict):
        new_graph_dict = {}
        count = 1
        graph_dict_size = len(graph_dict)
        for key, graph in graph_dict.items():
            print(f"Currently on graph {count} out of {graph_dict_size}")
            new_split_graph = self.split_graph(graph)
            connected_components = new_split_graph.connected_components(mode='WEAK')
            new_graphs = connected_components.subgraphs()
            count = count + 1

            for graph in new_graphs:
                main_show = self.determine_main_show(graph)
                new_graph_dict[main_show] = graph

        new_graph_dict = sorted(new_graph_dict.items(), key=lambda x: len(x[1].vs['name']),
                                reverse=True)

        new_graph_dict = {x[0]: x[1] for x in new_graph_dict}
        return new_graph_dict

    def determine_main_show(self, G):
        """The main show for each graph will be the one with the most members (to avoid graphs
        having a "Attack on Titan : Snickers Collab" key)"""
        members_of_each_show = self.anime_db.get_stats_of_shows(G.vs['name'], ['Scores'])
        return max(members_of_each_show, key=lambda x: members_of_each_show[x]['Scores'])

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

    def create_graph_of_all_anime(self):  # Main function

        self._all_graphs = {}

        def traverse_anime_relations(anime_id: int, G: Graph):

            def find_relevant_graph(t: str):  # Put this outside?
                for key, graph in self._all_graphs.items():
                    if t in graph.vs['name']:
                        return key, graph

            def inplace_union(G1, G2):  # Put this outside?
                for vertex in G2.vs:
                    if vertex['name'] not in G1.vs['name']:
                        G1.add_vertex(name=vertex['name'])

                for vertex in G2.vs:
                    for adjacent_vertex in vertex.neighbors():
                        if G2.are_connected(vertex, adjacent_vertex):
                            edge_id = G2.get_eid(vertex, adjacent_vertex)
                            edge = G2.es[edge_id]
                            G1.add_edge(vertex['name'], adjacent_vertex['name'], relation=edge['relation'])

            url = f'https://api.myanimelist.net/v2/anime/{int(anime_id)}?fields=id,media_type,mean,' \
                  f'related_anime,genres,' \
                  f'average_episode_duration,num_episodes,num_scoring_users'

            fail_count = 0
            # current_anime = call_function_through_process(get_search_results, url)
            current_anime = None
            while not current_anime:
                try:
                    time.sleep(1.2)
                    current_anime = get_search_results(url)
                except (TimeoutError, requests.exceptions.ReadTimeout) as e:
                    logger.debug(e)
                    current_anime = None

            # Add filter for getting none because 404 garbage
            while current_anime is None:  # turn this into decorator later, also change the None to not what we want
                fail_count += 1
                logger.warning("Anime was returned as None, sleeping and retrying")
                print(current_anime)
                time.sleep(30)  # Just in case
                print("Anime was returned as None, sleeping and retrying")
                # current_anime = call_function_through_process(get_search_results, url)
                current_anime = get_search_results(url)
                if fail_count == 10:
                    print("Failed to fetch resource 10 times in a row")
                    logger.debug("Failed to fetch resource 10 times in a row")
                    time.sleep(1800)

            title = current_anime['title']
            print(Fore.LIGHTWHITE_EX + f"Currently on title {title}")
            G.add_vertex(name=title)
            processed_anime.append(title)

            # Until here we can't know whether the show is related to any previous ones

            for related_anime in current_anime['related_anime']:
                related_title = related_anime['node']['title']
                if related_title not in relevant_titles:
                    continue
                relation_type = related_anime['relation_type']

                if related_title not in processed_anime:
                    traverse_anime_relations(related_anime['node']['id'], G)

                if related_title in G.vs['name']:  # If traverse was activated before (first time processing the anime)
                    # then the vertex should definitely be in the graph. Otherwise, this isn't the first time.
                    G.add_edge(title, related_title, relation=relation_type)
                else:
                    root, G_existing = find_relevant_graph(related_title)
                    if "delet dis" in G.vs['name']:
                        root = self.get_adjacent_vertex_names(G, "delet dis")[0]
                        G.delete_vertices("delet dis")

                    G_existing.add_vertex(title)
                    G_existing.add_edge(title, related_title, relation=relation_type)

                    inplace_union(G, G_existing)
                    self._all_graphs[root] = G.copy()

                    # maybe problem is here - check ID of G and graph_dict[root] after the eq, maybe use copy()
                    G.add_vertex(name="delet dis")
                    if root not in G.vs['name']:
                        G.add_vertex(name=root)
                    G.add_edge("delet dis", root)

                    print(f"Deleting graph. Graph's vertices are {G.vs['name']}")
                    logger.debug(f"Deleting graph. Graph's vertices are {G.vs['name']}")

        # ------------------ Main function starts here ------------------

        processed_anime = []

        partial_df_dict = self.anime_db.partial_df.to_dict(as_series=False)
        del partial_df_dict['Rows']

        relevant_ids_years_seasons_and_titles = [(show_stats[self.anime_db.stats["ID"]],
                                                  show_stats[self.anime_db.stats["Year"]],
                                                  show_stats[self.anime_db.stats["Season"]], title)
                                                 for title, show_stats in partial_df_dict.items()]

        relevant_ids_titles = [(x[0], x[3]) for x in sorted(relevant_ids_years_seasons_and_titles,
                                                            key=lambda x: (x[1], x[2]))]

        relevant_titles = [x[1] for x in relevant_ids_titles]

        for ID, MAL_title in relevant_ids_titles:

            if MAL_title in processed_anime:  # or MAL_title not in relevant_titles:
                continue

            G = Graph(directed=True)
            traverse_anime_relations(ID, G)

            if 'delet dis' not in G.vs['name']:
                self._all_graphs[MAL_title] = G
            else:
                # This handles a very specific case, need to elaborate later
                root = self.get_adjacent_vertex_names(G, "delet dis")[0]
                G.delete_vertices("delet dis")
                self._all_graphs[root] = G

            if len(self._all_graphs) % 20 == 0:
                save_pickled_file("test_graph_final3.pickle", self._all_graphs)

        self._all_graphs = self.split_graphs(self._all_graphs)

        save_pickled_file(graphs_dict_filename, self._all_graphs)


