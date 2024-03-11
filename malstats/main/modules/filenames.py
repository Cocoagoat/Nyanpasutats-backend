from pathlib import Path
current_dir = Path(__file__).parent
data_path = current_dir.parent / 'data'
logging_path = current_dir.parent / 'logging'
models_path = current_dir.parent / 'MLmodels'
model_filename_suffix = "RSDD"
current_model_name = "T1-4-50-RSDD.h5"
# current_model_filepath = models_path / current_model_name
auth_filename = data_path / "Authorization.txt"
user_database_name = data_path / "UserDB.parquet"
anime_database_name = data_path / "AnimeDB-T.parquet"
MAL_users_filename = data_path / "MALUsers.csv"
blacklist_users_filename = data_path / "Blacklist.csv"
scores_dict_filename = data_path / "ScoresDBDict.pickle"
aff_db_path = data_path / "Partials"
aff_db_filename = "AffinityDB"
# The lack of extension is on purpose ^
entry_tags_filename = data_path / "entry_tags_dict.pickle"
entry_tags_filename2 = data_path / "entry_tags_dict101.pickle"
shows_tags_filename = data_path / "shows_tags_dict.pickle"
graphs_dict_filename = data_path / "all_graphs.pickle"
relations_dict_filename = data_path / "anime_relations.pickle"
user_tag_affinity_dict_filename = data_path / "user_tag_affinity_dict.pickle"
# The below filenames are related to the creation of AffinityDB. Several variables which are
# normally taken from the various databases are saved as pickled files instead to avoid loading
# the heavy databases just for them.
relevant_shows_filename = data_path / "relevant_shows.pickle"
mean_score_per_show_filename = data_path / "mean_score_per_show.pickle"
scored_members_per_show_filename = data_path / "scored_members_per_show.pickle"
scored_shows_per_user_filename = data_path / "scored_shows_per_user.pickle"

# auth_filename = "Authorization.txt"
# user_database_name = "UserDB.parquet"
# anime_database_name = "AnimeDB.parquet"
# MAL_users_filename = "MALUsers.csv"
# blacklist_users_filename = "Blacklist.csv"
# scores_dict_filename = "ScoresDBDict.pickle"
# aff_db_path = data_path / "Partials"
# aff_db_filename = "AffinityDB-T5"
# # The lack of extension is on purpose ^
# entry_tags_filename = "entry_tags_dict2.pickle"
# entry_tags_filename2 = "entry_tags_dict2-2.pickle"
# shows_tags_filename = "shows_tags_dict2.pickle"
# graphs_dict_filename = "all_graphs.pickle"
# relations_dict_filename = "anime_relations.pickle"
# user_tag_affinity_dict_filename = "user_tag_affinity_dict.pickle"
# # The below filenames are related to the creation of AffinityDB. Several variables which are
# # normally taken from the various databases are saved as pickled files instead to avoid loading
# # the heavy databases just for them.
# relevant_shows_filename = "relevant_shows.pickle"
# mean_score_per_show_filename = "mean_score_per_show.pickle"
# scored_members_per_show_filename = "scored_members_per_show.pickle"
# scored_shows_per_user_filename = "scored_shows_per_user.pickle"

