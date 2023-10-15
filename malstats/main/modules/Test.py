from AffinityDB import AffinityDB
from UserDB import UserDB
from AnimeDB import AnimeDB
import random
from Tags import Tags
from filenames import *
from general_utils import *


if __name__ == '__main__':
    # print(5)
    # random_sums = [sum([random.randint(0, 1) for i in range(10000)]) for j in range(10000)]
    # print("Min:", min(random_sums))
    # print("Max:", max(random_sums))
    aff_db = AffinityDB()
    # affinity_db = aff_db.load_affinity_DB()
    # remove_zero_columns(affinity_db)
    # affinity_db.write_parquet(f"{aff_db_filename}.parquet")

    tags = Tags()
    k = tags.show_tags_dict
    k = aff_db.df
    print(5)

