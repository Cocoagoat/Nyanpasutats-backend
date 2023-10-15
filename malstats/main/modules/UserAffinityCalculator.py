from .User import User
from .GeneralData import GeneralData
from .Tags import Tags
import numpy as np


class UserAffinityCalculator:

    def __init__(self, user: User, data: GeneralData, tags: Tags):
        self.user = user
        self.data = data
        self.tags = tags

    def initialize_user_dicts(self):
        self.user.tag_affinity_dict = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.tag_pos_affinity_dict = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.score_per_tag = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.MAL_score_per_tag = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.tag_entry_list = {tag_name: [] for tag_name in self.tags.all_anilist_tags}
        self.user.freq_coeff_per_tag = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}
        self.user.tag_counts = {tag_name: 0 for tag_name in self.tags.all_anilist_tags}

    @staticmethod
    def initialize_user_stats(user, data):
        """Initializes the necessary stats for user's affinities to be calculated."""
        watched_user_score_list = [score for title, score in user.scores.items() if score]
        # ^ Simply a list of the user's scores for every show they watched

        watched_titles = [title for title, score in user.scores.items() if score]
        # watched_MAL_score_dict = self.data.mean_score_per_show.select(watched_titles).to_dict(as_series=False)
        watched_MAL_score_dict = data.mean_score_per_show.select(watched_titles).to_dict(as_series=False)
        watched_MAL_score_list = [score[0] for title, score in
                                  watched_MAL_score_dict.items() if title != "Rows"]

        user.MAL_mean_of_watched = np.mean(watched_MAL_score_list)
        user.mean_of_watched = np.mean(watched_user_score_list)
        user.std_of_watched = np.std(watched_user_score_list)
        user.entry_list = watched_titles
        return user

    def _calculate_affinities(self, tag):
        try:
            tag_overall_ratio = self.user.tag_counts[tag] / self.user.show_count
            freq_coeff = min(1, max(self.user.tag_counts[tag] / 10, tag_overall_ratio * 20))
            if "<" in tag:
                self.user.sum_of_freq_coeffs_dt += freq_coeff
            else:
                self.user.sum_of_freq_coeffs_st += freq_coeff
            self.user.freq_coeff_per_tag[tag] = freq_coeff
            # User has to watch either at least 10 shows with the tag or have the tag
            # in at least 5% of their watched shows for it to count fully.
            user_tag_diff = self.user.score_per_tag[tag] / self.user.tag_counts[tag] - self.user.mean_of_watched
            MAL_tag_diff = self.user.MAL_score_per_tag[tag] / self.user.tag_counts[tag] - self.user.MAL_mean_of_watched
            self.user.tag_affinity_dict[tag] = (2 * user_tag_diff - MAL_tag_diff) * freq_coeff
            self.user.tag_pos_affinity_dict[tag] = (self.user.tag_pos_affinity_dict[tag] / self.user.tag_counts[tag]) \
                                                   * freq_coeff

        # Add freq_coeff for each tag somewhere
        except ZeroDivisionError:
            self.user.tag_affinity_dict[tag] = 0
            self.user.tag_pos_affinity_dict[tag] = 0 # ?

    def _center_mean_of_affinities(self):
        def center_tags(tag_type):
            if tag_type == 'Single':
                tags = self.tags.all_single_tags + self.tags.all_genres + self.tags.all_studios
                freq_multi = len(tags) / self.user.sum_of_freq_coeffs_st
                mean_of_affs = np.mean([value for key, value in self.user.tag_affinity_dict.items()
                                        if "<" not in key])
                self.user.mean_of_single_affs = mean_of_affs
                self.user.freq_multi_st = freq_multi
            else:
                tags = self.tags.all_doubletags
                freq_multi = len(tags) / self.user.sum_of_freq_coeffs_dt
                mean_of_affs = np.mean([value for key, value in self.user.tag_affinity_dict.items()
                                        if "<" in key])
                self.user.mean_of_double_affs = mean_of_affs
                self.user.freq_multi_dt = freq_multi

            for tag in tags:
                self.user.tag_affinity_dict[tag] = self.user.tag_affinity_dict[tag] - mean_of_affs\
                                                   * freq_multi * self.user.freq_coeff_per_tag[tag]

        if self.data.OG_aff_means:
            self.user.center_tag_aff_dict_means(self.data.OG_aff_means)
        center_tags(tag_type='Single')
        center_tags(tag_type='Double')

    def _process_entry_tags(self, related_entry, length_coeff, user_score, MAL_score):

        MAL_score_coeff = -0.6 * MAL_score + 5.9
        related_entry_data = self.tags.entry_tags_dict[related_entry]
        for tag in related_entry_data['Tags'] + related_entry_data['DoubleTags']:
            if tag['name'] not in self.tags.all_anilist_tags:
                continue
            adjusted_p = tag['percentage'] # fix later
            # # On Anilist, every tag a show has is listed with a percentage. This percentage
            # # can be pushed down or up a bit by any registered user, so as a variable it
            # # inevitably introduces human error - a tag with 60% means that not all users agree
            # # that said tag is strongly relevant to the show, and thus we want to reduce its weight
            # # by more than just 40% compared to a 100% tag.

            self.user.score_per_tag[tag['name']] += user_score * adjusted_p * length_coeff
            self.user.MAL_score_per_tag[tag['name']] += MAL_score * adjusted_p * length_coeff
            self.user.tag_counts[tag['name']] += adjusted_p * length_coeff
            self.user.tag_entry_list[tag['name']].append(related_entry)

            if user_score >= self.user.mean_of_watched:
                self.user.tag_pos_affinity_dict[tag['name']] += MAL_score_coeff * \
                                                                self.user.exp_coeffs[10 - user_score] \
                                                                * adjusted_p * length_coeff

        # for genre in self.tags.entry_tags_dict[related_entry]['Genres']:
        for genre in related_entry_data['Genres']:
            self.user.score_per_tag[genre] += user_score * length_coeff
            self.user.MAL_score_per_tag[genre] += MAL_score * length_coeff
            self.user.tag_counts[genre] += length_coeff
            self.user.tag_entry_list[genre].append(related_entry)
            # Genres and studios do not have percentages, so we add 1 as if p=100

            if user_score >= self.user.mean_of_watched:
                self.user.tag_pos_affinity_dict[genre] += MAL_score_coeff * \
                                                          self.user.exp_coeffs[10 - user_score] \
                                                          * length_coeff

        show_studio = self.tags.entry_tags_dict[related_entry]['Studio']
        if show_studio in self.tags.all_anilist_tags:
            self.user.score_per_tag[show_studio] += user_score * length_coeff
            self.user.MAL_score_per_tag[show_studio] += MAL_score * length_coeff
            self.user.tag_counts[show_studio] += length_coeff
            self.user.tag_entry_list[show_studio].append(related_entry)

            if user_score >= self.user.mean_of_watched:
                self.user.tag_pos_affinity_dict[show_studio] += MAL_score_coeff * \
                                                                self.user.exp_coeffs[10 - user_score] \
                                                                * length_coeff

    def get_user_affs(self):

        processed_entries = []
        self.user.entry_list = []
        self.user.show_count = 0

        self.initialize_user_dicts()
        # self.get_user_list_stats()
        self.user = self.initialize_user_stats(self.user, self.data)

        try:
            x = int(np.ceil(self.user.mean_of_watched))
        except ValueError:
            # This means the user has no watched shows, or there is another issue with the data.
            print(f"Mean is {self.user.mean_of_watched}")
            return self.user

        self.user.score_diffs = [score - self.user.mean_of_watched for score in range(10, x - 1, -1)]
        self.user.exp_coeffs = [1 / 2 ** (10 - exp) * self.user.score_diffs[10 - exp]
                                for exp in range(10, x - 1, -1)]

        for entry in self.user.entry_list:
            user_score = self.user.scores[entry]
            if not user_score or entry in processed_entries:
                continue

            main_show = self.tags.entry_tags_dict[entry]['Main']
            main_show_data = self.tags.show_tags_dict[main_show]
            user_watched_entries_length_coeffs = [x[1] for x in
                                                  main_show_data['Related'].items()
                                                  if self.user.scores[x[0]]]

            sum_of_length_coeffs = sum(user_watched_entries_length_coeffs)

            for related_entry, length_coeff in main_show_data['Related'].items():
                processed_entries.append(related_entry)
                user_score = self.user.scores[related_entry]
                if not user_score:
                    continue

                length_coeff = length_coeff / sum_of_length_coeffs

                MAL_score = self.data.mean_score_per_show[related_entry][0]
                if MAL_score < 6.5:
                    print("Warning - MAL score lower than 6.5")
                    continue

                self.user.show_count += length_coeff

                # Calculates sums (user_score_per_show, MAL_score_per_show, etc)
                self._process_entry_tags(related_entry, length_coeff, user_score, MAL_score)

        for tag in self.tags.all_anilist_tags:
            self._calculate_affinities(tag)

        self._center_mean_of_affinities()

        return self.user

    def recalc_affinities_2(self, entry):

        main_show = entry
        sum_of_length_coeffs = sum([length_coeff for related_entry, length_coeff
                                    in self.tags.show_tags_dict[main_show]['Related'].items()
                                    if self.user.scores[related_entry]])

        if not sum_of_length_coeffs:
            return

        new_user_score_per_tag = self.user.score_per_tag.copy()
        new_user_tag_count = self.user.tag_counts.copy()
        new_user_show_count = self.user.show_count
        new_MAL_score_per_tag = self.user.MAL_score_per_tag.copy()
        new_freq_coeffs = self.user.freq_coeff_per_tag.copy()
        new_mean_of_watched = self.user.mean_of_watched
        new_MAL_mean = self.user.MAL_mean_of_watched
        pos_aff_per_entry = {}
        show_tag_names = set()

        for entry, length_coeff in self.tags.show_tags_dict[main_show]['Related'].items():
            length_coeff /= sum_of_length_coeffs
            entry_user_score = self.user.scores[entry]
            if not entry_user_score:
                continue
            entry_MAL_score = self.data.mean_score_per_show[entry].item()

            new_mean_of_watched = (new_mean_of_watched * new_user_show_count) - entry_user_score * length_coeff
            new_MAL_mean = (new_MAL_mean * new_user_show_count) - entry_MAL_score * length_coeff

            new_user_show_count -= length_coeff

            new_mean_of_watched /= new_user_show_count
            new_MAL_mean /= new_user_show_count

            entry_MAL_coeff = self.MAL_score_coeff(entry_MAL_score)
            # Turn the data thingy into a dict so no .item()
            try:
                entry_exp_coeff = self.user.exp_coeffs[10 - entry_user_score]
            except IndexError:
                entry_exp_coeff = 0

            pos_aff_per_entry[entry] = entry_MAL_coeff * entry_exp_coeff * length_coeff
            entry_tags_dict = self.tags.entry_tags_dict2[entry]

            for tag, tag_p in entry_tags_dict.items():

                try:
                    tag_p = tag_p['percentage']
                except KeyError:
                    tag_p = 1

                try:
                    # tag_overall_ratio =  (self.user.tag_counts[tag]-length_coeff)/(self.user.show_count-length_coeff)\
                    new_user_tag_count[tag] -= length_coeff * tag_p
                    try:
                        tag_overall_ratio = new_user_tag_count[tag]/new_user_show_count
                    except ZeroDivisionError:
                        return
                        # This is an EXTREMELY rare case where a person has only ONE relevant scored show.
                        # Not going to bother handling it properly, since the requirement to be in the database
                        # is 50 scored shows - for exactly 49 of them to be irrelevant (too obscure) is nigh impossible.
                        # The one person out of the millions tested who triggered this was someone who watches almost
                        # exclusively hentai, and their one relevant scored show was Keijo!!!!!!!!. Why man, just why.

                    # tag_freq_coeff = min(1, max(new_user_tag_count[tag] / 10, tag_overall_ratio * 20))
                    new_freq_coeffs[tag] = min(1, max(new_user_tag_count[tag] / 10, tag_overall_ratio * 20))
                except KeyError:
                    # Tag is a studio that appears in entry_tags_dict
                    # but isn't in tags.all_anilist_tags (too niche)
                    continue

                show_tag_names.add(tag)
                new_user_score_per_tag[tag] -= entry_user_score * tag_p * length_coeff
                new_MAL_score_per_tag[tag] -= entry_MAL_score * tag_p * length_coeff
                # user_score_per_tag = self.user.score_per_tag[tag] - entry_user_score * tag_p
                # user_tag_count = self.user.tag_counts[tag] - tag_p

        for tag in show_tag_names:
            pos_aff_per_entry_for_tag = pos_aff_per_entry.copy()
        # for entry, length_coeff in self.tags.show_tags_dict[main_show]['Related'].items():
        #     entry_tags_dict = self.tags.entry_tags_dict2[entry]
        #     for tag, tag_p in entry_tags_dict.items():
            try:
                user_tag_diff = new_user_score_per_tag[tag] / new_user_tag_count[tag] - new_mean_of_watched
            except ZeroDivisionError:
                self.user.adj_tag_affinity_dict[tag] = 0
                self.user.adj_pos_tag_affinity_dict[tag] = 0
                continue  # This entry was the only one with the tag, so affinities are 0 without it

            # MAL_score_per_tag = self.user.MAL_score_per_tag[tag] - entry_MAL_score * tag_p
            MAL_tag_diff = new_MAL_score_per_tag[tag] / new_user_tag_count[tag] - new_MAL_mean

            self.user.adj_tag_affinity_dict[tag] = (2 * user_tag_diff - MAL_tag_diff) * new_freq_coeffs[tag]
            self.user.adj_tag_affinity_dict[tag] -= self.data.OG_aff_means[tag]

            mean_of_affs = self.user.mean_of_double_affs if "<" in tag else self.user.mean_of_single_affs
            freq_multi = self.user.freq_multi_dt if "<" in tag else self.user.freq_multi_st
            self.user.adj_tag_affinity_dict[tag] -= mean_of_affs * freq_multi * new_freq_coeffs[tag]

            # entry_pos_aff /= self.user.tag_counts[tag] # do minus here?
            # entry_pos_aff *= self.user.freq_coeff_per_tag[tag]
            for entry in pos_aff_per_entry.keys():
                try:
                    pos_aff_per_entry_for_tag[entry] *= self.tags.entry_tags_dict2[entry][tag]['percentage']
                except KeyError:
                    pos_aff_per_entry_for_tag[entry] = 0
            show_pos_aff = sum(pos_aff_per_entry_for_tag.values())

            OG_pos_aff = self.user.tag_pos_affinity_dict[tag] * self.user.tag_counts[tag]
            OG_pos_aff /= self.user.freq_coeff_per_tag[tag]
            new_pos_aff = OG_pos_aff - show_pos_aff
            new_pos_aff *= new_freq_coeffs[tag]
            new_pos_aff /= new_user_tag_count[tag]
            self.user.adj_pos_tag_affinity_dict[tag] = new_pos_aff

    @staticmethod
    def MAL_score_coeff(score):
        return -0.6 * score + 5.9