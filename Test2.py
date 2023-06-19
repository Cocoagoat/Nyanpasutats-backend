from Test import *
from MAL_utils import Seasons
import requests
from MAL_utils import Data
from polars.exceptions import ColumnNotFoundError
import igraph as ig
from igraph import Graph, summary, union
from typing import Dict, Set
from igraph.drawing import Plot
import inspect
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(igraph_graph):
    labels_list = igraph_graph.vs['name']
    node_labels = {i: label for i, label in enumerate(labels_list)}
    # Convert igraph graph to NetworkX graph
    networkx_graph = igraph_graph.to_networkx()
    edge_labels = nx.get_edge_attributes(networkx_graph, 'relation')

    # Draw the NetworkX graph using matplotlib
    pos = nx.spring_layout(networkx_graph)
    plt.figure(figsize=(51.2,28.8))
    nx.draw_networkx_nodes(networkx_graph, pos, node_size=200, node_color='skyblue')
    nx.draw_networkx_labels(networkx_graph, pos, labels=node_labels, font_size=6)
    nx.draw_networkx_edges(networkx_graph, pos, edgelist=networkx_graph.edges, edge_color='gray',
                           connectionstyle=f"arc{3},rad={0.1}")
    for edge, label in nx.get_edge_attributes(networkx_graph, 'relation').items():
        start, end = pos[edge[0]], pos[edge[1]]
        mid = (start + end) / 2
        label_pos = mid + 0.1 * (end - start)[::-1] * [-1, 1] * (hash(edge) % 2 * 2 - 1)
        plt.text(label_pos[0], label_pos[1], label, horizontalalignment='center', verticalalignment='center',
                 fontsize=4, weight='bold', color ='blue')

    # Show the plot
    plt.show()


def get_adjacent_vertex_names(G, vertex_name):
    adj_vertex_list=[]
    incident_edges = G.incident(vertex_name)
    for edge_index in incident_edges:
        edge = G.es[edge_index]
        source_vertex_index, target_vertex_index = edge.tuple
        adj_vertex = G.vs[target_vertex_index]['name']
        adj_vertex_list.append(adj_vertex)
    return adj_vertex_list


@timeit
def get_stats_of_shows(show_list, relevant_stats):
    """ Will create a dictionary that has every show in show_list as the key, and every stat in relevant_stats
        in a list as the value.
        Example :
        {'Shingeki no Kyojin': {'ID': 16498.0, 'Mean Score': 8.53, 'Members': 3717089.0},
         'Shingeki no Kyojin OVA': {'ID': 18397.0, 'Mean Score': 7.87, 'Members': 439454.0}"""

    stats_dict = {}
    for show in show_list:
        show_dict = {}
        for stat in relevant_stats:
            try:
                show_dict[stat] = data.anime_df.filter(pl.col('Rows') == stat)[show].item()
            except ColumnNotFoundError:
                break
            except ValueError:
                show_dict[stat] = None
        if show_dict:
            stats_dict[show] = show_dict
    return stats_dict


def are_separate_shows(show1,show2, relation_type):



    def both_shows_are_TV():
        return show_stats[show1]["Type"] == 1 and show_stats[show2]["Type"] == 1

    def show_is_longer_than(minutes,name):
        if not show_stats[name]["Episodes"]:
            show_stats[name]["Episodes"] = 1
        # if show_stats[name]["Duration"]==1:
        #     show_stats[name]["Duration"]=65
        return show_stats[name]["Duration"] * show_stats[name]["Episodes"] > minutes

    if show1 not in data.titles or show2 not in data.titles: #take care of this outside later
        return True

    relevant_stats = ["Duration", "Episodes", "Type"]
    show_stats = get_stats_of_shows([show1,show2], relevant_stats)
    #Put these into the 3rd case^
    match relation_type:
        case 'sequel' | 'prequel' | 'alternative_version' | 'summary':
            return False
        case 'character':
            return True
        case 'other' | 'side_story' | 'spin_off':
            if both_shows_are_TV() or (show_is_longer_than(180,show1) and show_is_longer_than(180,show2)):
                return True
            return False
        case 'alternative_setting':
            if (show_is_longer_than(60, show1) and show_is_longer_than(60, show2)):
                return True
            return False
        case _:
            # print(f"Warning, separate case, relation type is {relation_type}")
            return False


# def get_opposite_edge(G,edge):
#     opposite_edge_ID = G.get_eid(edge.target_vertex, edge.source_vertex, error=False)
#     return G.es[opposite_edge_ID]


def manually_separate_shows(edges_to_delete):
    """ Adds extra edges to delete that couldn't be identified by the algorithm in split_graph.
        MAL's database is extremely inconsistent, and has things like Love Live! Sunshine being listed
        as a sequel of Love Live, despite Nijigasaki and Superstar being listed as alternative_setting.
        pass. Normally a sequel would be part of the original show, so there is no way to detect that."""


def split_graph(G):
    G=G.copy()
    edges_to_delete=[]
    for vertex in G.vs:
        vertex_edge_list = vertex.incident()
        for edge in vertex_edge_list:
            # print(f"Current edge : {edge}")
            v1,v2 = edge.vertex_tuple

            relation_type = edge['relation']
            if are_separate_shows(v1['name'],v2['name'],relation_type):
                edges_to_delete.append((v1.index,v2.index))
                edges_to_delete.append((v2.index,v1.index))


    # manually_separate_shows(edges_to_delete)

    for v1,v2 in set(edges_to_delete):
        edge_to_delete = G.get_eid(v1, v2, directed=True, error=False)
        if edge_to_delete!=-1:
            G.delete_edges(edge_to_delete)
    return G


def split_graphs(graph_dict):
    new_graph_dict={}
    count=1
    graph_dict_size = len(graph_dict)
    for key, graph in graph_dict.items():
        print(f"Currently on graph {count} out of {graph_dict_size}")
        new_split_graph = split_graph(graph)
        connected_components = new_split_graph.connected_components(mode='WEAK')
        new_graphs = connected_components.subgraphs()
        count=count+1

        for graph in new_graphs:
            main_show = determine_main_show(graph)
            new_graph_dict[main_show] = graph
    return new_graph_dict



def determine_main_show(G):
    members_of_each_show = get_stats_of_shows(G.vs['name'],['Scores'])
    return max(members_of_each_show, key=lambda x: members_of_each_show[x]['Scores'])


def graph_dict_to_database(graph_dict):

    # This part properly renames the roots of each graph to the main show (to avoid situations such as
    # a random PV from 2018 for a 2020 show being considered the root node because it aired earlier)

    # Do we need this if split_graphs already does it?
    # original_dict = graph_dict.items()
    # for key,graph in original_dict:
    #     main_show = determine_main_show(graph)
    #     if key!=main_show:
    #         graph_dict.remove(key)
    #         graph_dict[main_show] = graph

    # This part creates a dictionary of all main shows as keys and their related shows (including themselves)
    # as the values, for example
    # {'Shingeki no Kyojin' : ['Shingeki no Kyojin', 'Shingeki no Kyojin Season 2', ....], 'Steins;Gate : [...]}
    related_dict ={}
    for key,graph in graph_dict.items():
        related_dict[key] = get_stats_of_shows(graph.vs['name'],data.anime_db_stats)

    return related_dict





def create_graph_of_all_anime(): #Main function
    def traverse_anime_relations(anime_id: int, G: Graph):

        def find_relevant_graph(t):  # Put this outside?
            for key, graph in graph_dict.items():
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
        current_anime = call_function_through_process(get_search_results, url)
        time.sleep(1)
        while current_anime is None:  # turn this into decorator later, also change the None to not what we want
            fail_count += 1
            logger.warning("Anime was returned as None, sleeping and retrying")
            print(current_anime)
            time.sleep(30)  # Just in case
            print("Anime was returned as None, sleeping and retrying")
            current_anime = call_function_through_process(get_search_results, url)
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
                    root = get_adjacent_vertex_names(G, "delet dis")[0]
                    G.delete_vertices("delet dis")

                G_existing.add_vertex(title)
                G_existing.add_edge(title, related_title, relation=relation_type)

                inplace_union(G, G_existing)
                graph_dict[root] = G.copy()

                # maybe problem is here - check ID of G and graph_dict[root] after the eq, maybe use copy()
                G.add_vertex(name="delet dis")
                if root not in G.vs['name']:
                    G.add_vertex(name=root)
                G.add_edge("delet dis", root)

                print(f"Deleting graph. Graph's vertices are {G.vs['name']}")
                logger.debug(f"Deleting graph. Graph's vertices are {G.vs['name']}")

    def show_meets_conditions(show_stats):
        if show_stats["Scores"] and show_stats["Scores"] >= 2000 \
                and show_stats["Duration"] * show_stats["Episodes"] >= 15:
            return True
        return False

# ------------------ Main function starts here ------------------
    prompt = input("Use old graphs list? Y/N")

    if prompt == 'Y':
        graph_dict = load_pickled_file('test_graph5.pickle')
        # graph_dict.pop('Bokura no Live Kimi to no Life')
        processed_anime = [title for key, graph in graph_dict.items() for title in graph.vs['name']]
    else:
        graph_dict = {}
        processed_anime = []

    # ids = data.anime_df.row(0)[1:]
    # release_years = data.anime_df.row(8)[1:]
    # seasons = data.anime_df.row(9)[1:]
    # scored_shows = data.anime_df.row(2)[1:]

    stats_dict = get_stats_of_shows(data.titles,["ID", "Year", "Season", "Scored", "Duration", "Episodes"])

    relevant_ids_and_titles = [(show_stats["ID"],title) for title,show_stats in stats_dict.items()
                               if title not in data.unlisted_titles and show_meets_conditions(show_stats)]

    # ids_titles_years_seasons_scored = [x for x in list(zip(ids, data.titles, release_years, seasons, scored_shows))
    #                                    if x[2] is not None and x[4] is not None and x[4] > 2000 and x[4]]
    #
    # ids_titles_years_seasons_sorted = sorted(ids_titles_years_seasons_scored, key=lambda x: (x[2], x[3], x[0]))
    # List of anime entries, each containing ID, title, airing_season, airing_year, and scored_amount

    # relevant_titles = [x[1] for x in ids_titles_years_seasons_sorted if x[4] is not None and x[4] > 2000
    #                    and x[1] not in data.unlisted_titles]


    # for ID, MAL_title, year, season, _ in ids_titles_years_seasons_sorted:
    for ID, MAL_title in relevant_ids_and_titles:

        k=5

        if MAL_title in processed_anime: # or MAL_title not in relevant_titles:
            continue

        G = Graph(directed=True)

        traverse_anime_relations(ID, G)

        if 'delet dis' not in G.vs['name']:
            graph_dict[MAL_title] = G
        else:
            # This handles a very specific case, need to elaborate later
            root = get_adjacent_vertex_names(G, "delet dis")[0]
            G.delete_vertices("delet dis")
            graph_dict[root] = G

        if len(graph_dict) % 20 == 0:
            save_pickled_file("test_graph_final.pickle", graph_dict)

    save_pickled_file("test_graph_final.pickle", graph_dict)

    split_graph_dict = split_graphs(graph_dict)

    save_pickled_file("split_graphs_final.pickle", split_graph_dict)

    related_dict = {}
    for key, graph in split_graph_dict.items():
        related_dict[key] = get_stats_of_shows(graph.vs['name'], data.anime_db_stats)

    save_pickled_file("related_dict_final.pickle", related_dict)


    # Example usage: Access and print/plot the graph by title
    # title_to_find = 'K-On!!'
    # graph_to_print = graph_dict[title_to_find]
    # summary(graph_to_print)


if __name__ == '__main__':
    data=Data()
    # url = 'https://api.myanimelist.net/v2/anime/ranking?ranking_type=all&limit=100&' \
    #       'fields=mean,num_scoring_users,num_list_users,num_episodes,average_episode_duration,' \
    #       'media_type,start_season'
    # response = get_search_results(url)
    # url = 'https://api.myanimelist.net/v2/anime?q=one&limit=100'
    # response2 = get_search_results(url)
    print(5)
    data.generate_anime_DB()
    # test_graph = load_pickled_file("test_graph2.pickle")
    # index_graph = test_graph["Love Live! School Idol Project"]
    # test_split = split_graph(index_graph)
    # plot_graph(test_split)
    #

    # graph_dict = load_pickled_file("test_graph5.pickle")
    # split_graphs(graph_dict)
    # create_graph_of_all_anime()

