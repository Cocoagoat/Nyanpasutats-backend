user_database_name = "UserDB.parquet"
anime_database_name = "AnimeDB.parquet"
MAL_users_filename = "MALUsers.csv"
blacklist_users_filename = "Blacklist.csv"
scores_dict_filename = "ScoresDBDict.pickle"
aff_db_filename = "AffinityDB-T3"
# The lack of extension is on purpose ^
entry_tags_filename = "entry_tags_dict2.pickle"
shows_tags_filename = "shows_tags_dict2.pickle"
graphs_dict_filename = "all_graphs.pickle"
relations_dict_filename = "anime_relations.pickle"
user_tag_affinity_dict_filename = "user_tag_affinity_dict.pickle"
# The below filenames are related to the creation of AffinityDB. Several variables which are
# normally taken from the various databases are saved as pickled files instead to avoid loading
# the heavy databases just for them.
relevant_shows_filename = "relevant_shows.pickle"
mean_score_per_show_filename = "mean_score_per_show.pickle"
scored_members_per_show_filename = "scored_members_per_show.pickle"
scored_shows_per_user_filename = "scored_shows_per_user.pickle"

