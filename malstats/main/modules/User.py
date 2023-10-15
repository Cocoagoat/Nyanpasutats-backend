from dataclasses import dataclass, field


@dataclass
class User:
    tag_affinity_dict: dict = field(default=None)
    tag_pos_affinity_dict: dict = field(default=None)
    adj_tag_affinity_dict : dict = field(default=None)
    adj_pos_tag_affinity_dict : dict = field(default=None)

    mean_of_single_affs: int = field(default=None)
    mean_of_double_affs: int = field(default=None)

    name: str = field(default=None)
    scored_amount: int = field(default=None)
    scores: dict = field(default=None)

    mean_of_watched: float = field(default=None)
    std_of_watched: float = field(default=None)
    MAL_mean_of_watched: float = field(default=None)

    score_per_tag: dict = field(default=None)
    MAL_score_per_tag: dict = field(default=None)
    positive_aff_per_tag: dict = field(default=None)
    freq_coeff_per_tag: dict = field(default=None)
    tag_entry_list : dict = field(default=None)
    tag_counts: dict = field(default=None)

    sum_of_freq_coeffs_st: int = field(default=0)
    sum_of_freq_coeffs_dt: int = field(default=0)

    freq_multi_st : int = field(default=0)
    freq_multi_dt : int = field(default=0)

    entry_list: list = field(default=None)
    show_count: int = field(default=None)

    score_diffs: list = field(default=None)
    exp_coeffs: list = field(default=None)

    def center_tag_aff_dict_means(self, means):
        for tag in self.tag_affinity_dict:
            # Normalizing the affinities to each tag, with the means being calculated in generate_data().
            try:
                self.tag_affinity_dict[tag] =\
                    self.tag_affinity_dict[tag] - means[tag]
            except TypeError:
                continue