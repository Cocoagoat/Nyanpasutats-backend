# def get_opposite_edge(G,edge):
#     opposite_edge_ID = G.get_eid(edge.target_vertex, edge.source_vertex, error=False)
#     return G.es[opposite_edge_ID]

# def is_vertex_cut(g, vertices):
#     # Get the number of connected components in the original graph
#     original_components = len(g.clusters())
#
#     # Remove the vertices from the graph
#     g.delete_vertices(vertices)
#
#     # Get the number of connected components in the modified graph
#     modified_components = len(g.clusters())
#
#     # Return True if the number of connected components has increased, else False
#     return modified_components > original_components


# def cut_crossovers(G):
#     G_copy = G.copy()
#     for vertex in G.vs:
#         vertex_edge_list = vertex.incident()
#         cutoff_flag = True
#         for edge in vertex_edge_list:
#             if edge['relation'] not in ['character','other','side_story','spin_off']:
#                 cutoff_flag = False
#                 break
#         if cutoff_flag:
#             if is_vertex_cut(G,vertex):
#                 edge_copy = edge.copy()
#                 G.delete_vertices(vertex)
#                 connected_components = G.connected_components(mode='WEAK')
#                 new_graphs = connected_components.subgraphs()
#
#                 # new_graphs[0].add_vertex(vertex)
#                 # vertex_to_connect_to = edge.source if edge.source['name'] != vertex['name']\
#                 #                                     else edge.target
#                 # if edge.target in new_graphs[0]


# @staticmethod
# def graph_dict_to_database(graph_dict):
#     # This part properly renames the roots of each graph to the main show (to avoid situations such as
#     # a random PV from 2018 for a 2020 show being considered the root node because it aired earlier)
#
#     # This part creates a dictionary of all main shows as keys and their related shows (including themselves)
#     # as the values, for example
#     # {'Shingeki no Kyojin' : ['Shingeki no Kyojin', 'Shingeki no Kyojin Season 2', ....], 'Steins;Gate : [...]}
#     related_dict = {}
#     for key, graph in graph_dict.items():
#         related_dict[key] = anime_db.get_stats_of_shows(graph.vs['name'], anime_db.stats)
#
#     return related_dict