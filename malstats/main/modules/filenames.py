from pathlib import Path
current_dir = Path(__file__).parent
data_path = current_dir.parent / 'data'
logging_path = current_dir.parent / 'logging'
models_path = current_dir.parent / 'MLmodels'
model_filename_suffix = "RSDD"
main_model_path = models_path / "Main_prediction_model.h5"
auth_filename = data_path / "Authorization.txt"
user_database_name = data_path / "UserDB.parquet"
anime_database_name = data_path / "AnimeDB.parquet"
anime_database_updated_name = data_path / "AnimeDB-U.parquet"
anime_database_prev_updated_name = data_path / "AnimeDB-U-prev.parquet"
MAL_users_filename = data_path / "MALUsers.csv"
blacklist_users_filename = data_path / "Blacklist.csv"
scores_dict_filename = "ScoresDBDict.pickle"
scores_dict_filepath = data_path / scores_dict_filename
aff_db_path = data_path / "Partials"
aff_db_filename = "AffinityDB"
# The lack of extension is on purpose ^
entry_tags_filename = data_path / "entry_tags_dict.pickle"
entry_tags_nls_filename = data_path / "entry_tags_dict_nls.pickle"
entry_tags_updated_filename = data_path / "entry_tags_dict-U.pickle"
entry_tags_nls_updated_filename = data_path / "entry_tags_dict_nls-U.pickle"
entry_tags2_filename = data_path / "entry_tags_dict2.pickle"
shows_tags_filename = data_path / "shows_tags_dict.pickle"
shows_tags_nls_filename = data_path / "shows_tags_dict_nls.pickle"
shows_tags_updated_filename = data_path / "shows_tags_dict-U.pickle"
shows_tags_nls_updated_filename = data_path / "shows_tags_dict_nls-U.pickle"
temp_graphs_dict_filename = data_path / "temp_all_graphs.pickle"
graphs_dict_filename = data_path / "all_graphs.pickle"
graphs_dict_updated_filename = data_path / "all_graphs-U.pickle"
graphs_dict_nls_filename = data_path / "all_graphs_nls.pickle"
graphs_dict_nls_updated_filename = data_path / "all_graphs_nls-U.pickle"
relations_dict_filename = data_path / "anime_relations.pickle"
user_tag_affinity_dict_filename = data_path / "user_tag_affinity_dict.pickle"
# The below filenames are related to the creation of AffinityDB. Several variables which are
# normally taken from the various databases are saved as pickled files instead to avoid loading
# the heavy databases just for them.
relevant_shows_filename = data_path / "relevant_shows.pickle"
mean_score_per_show_filename = data_path / "mean_score_per_show.pickle"
scored_members_per_show_filename = data_path / "scored_members_per_show.pickle"
scored_shows_per_user_filename = data_path / "scored_shows_per_user.pickle"


