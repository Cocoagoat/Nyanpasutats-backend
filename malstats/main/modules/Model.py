import json
import os
import random
import numpy as np
import pandas as pd
import logging
from keras.regularizers import l1
from main.modules.filenames import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from main.modules.AffinityDB import AffinityDB, AffDBEntryCreator
from main.modules.AnimeDB import AnimeDB
from main.modules.GeneralData import GeneralData
from main.modules.Tags import Tags
from main.modules.User import User
from main.modules.UserDB import UserDB
from main.modules.general_utils import load_pickled_file
from main.models import AnimeDataUpdated

recs_logger = logging.getLogger("nyanpasutats.recs_logger")


class ModelParams:
    def __init__(self, layer_count, layer_sizes: list, algo: str,
                 lr, reg_rate, loss: str):
        if len(layer_sizes) != layer_count:
            raise ValueError("Layer sizes array must be the same length "
                             "as the amount of layers")

        if loss not in ['mean_squared_error', 'mean_absolute_error']:
            raise ValueError("Untested/unknown loss function")

        if algo not in ["Adam", "RMSprop", "SGD"]:
            raise ValueError("Untested/unknown optimization algorithm")

        if lr < 0:
            raise ValueError("A negative learning rate is not allowed")

        self.layer_count = layer_count
        self.layer_sizes = layer_sizes
        self.algo = algo
        self.lr = lr
        self.reg_rate = reg_rate
        self.loss = loss

    @staticmethod
    def random_init():
        layer_count = random.choice([3, 4])
        neuron_start = random.choice([1280, 1536, 2048])
        neuron_lower = [bool(round(random.randint(0, 1))) for _ in range(layer_count)]

        # The below line will create layer_sizes of [K, 1/2K, 1/4K, 1/4K]
        # for neuron_start = K and neuron_lower = [True, True, False], for example.
        layer_sizes = [neuron_start*(1/2)**len([k for k in neuron_lower[:i] if k]) for i in range(layer_count)]
        algo = random.choice(["RMSprop"])
        lr = random.choice([0.01, 0.02, 0.03])
        reg_rate = random.choice([0])
        loss = "mean_absolute_error"
        epoch_count = 50
        with open(data_path / "ModelTests" / f"temp_model_stats.txt", "a") as f:
            f.write(f"Reg_rate : {reg_rate}, lr : {lr}\n\n")
        return ModelParams(layer_count=layer_count, layer_sizes=layer_sizes, algo=algo, reg_rate=reg_rate,
                           lr=lr, loss=loss, epoch_count=epoch_count)


class Model:
    def __init__(self, model: tf.keras.Sequential):
        self.model = model

    def train(self, batch_size=2048, epochs=50, filename=None):

        def interleave(files):
            num_features = self.model.layers[0].input_shape[1]
            return files.interleave(lambda file_path: tf.data.Dataset.from_generator(
                _generate_batch, args=[file_path], output_signature=(
                    tf.TensorSpec(shape=(batch_size, num_features), dtype=tf.float32),
                    tf.TensorSpec(shape=(batch_size,), dtype=tf.int32))
            ), cycle_length=1)

        def get_interleaved_training_data():
            file_amount = AffinityDB.count_major_parts()
            file_paths = [str(aff_db_path / f"{aff_db_filename}-P{_}-N-{model_filename_suffix}.parquet") for _ in
                          range(1, file_amount+1)]

            files = tf.data.Dataset.list_files(file_paths)
            train_files = files.take(file_amount - 1)
            test_files = files.skip(file_amount - 1)

            train_dataset = interleave(train_files)
            test_dataset = interleave(test_files)

            return train_dataset, test_dataset

        def _generate_batch(file_path_tensor):
            # Read a single parquet file using PyArrow
            file_path = file_path_tensor.decode('utf-8')
            df = pd.read_parquet(file_path)
            labels = df['User Score']
            features = df.drop(columns=['User Score'])
            total_rows = len(df)

            # Iterating through the DataFrame in chunks of size batch_size
            for k in range(0, total_rows - batch_size, batch_size):
                batch_features = features.iloc[k: k + batch_size]
                batch_labels = labels.iloc[k: k + batch_size]
                yield batch_features, batch_labels

        train_dataset, test_dataset = get_interleaved_training_data()
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.0003)
        checkpoint_callback = ModelCheckpoint(
            filepath=main_model_path.parent / 'current_model_epoch_{epoch:02d}.h5',  # Save the model with epoch number in the filename
            save_freq='epoch'  # Save the model after every epoch
        )

        print("Beginning training")

        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            # Number of times the entire dataset is passed forward and backward through the neural network
            validation_data=test_dataset,
            verbose=1,  # To view the training progress for each epoch
            callbacks=[early_stop_callback, checkpoint_callback]
        )

        history_data = history.history

        filename = main_model_path if not filename else filename
        if not os.path.exists(filename):
            self.model.save(filename)
        else:
            for i in range(2, 10000):
                filename = main_model_path.parent / f"Main_prediction_model{i}.h5"
                if not os.path.exists(filename):
                    with open(data_path / f'training_history{i}.json', 'w') as f:
                        json.dump(history_data, f)
                    self.model.save(filename)
                    return
            raise FileExistsError("File already exists")  # Technically impossible unless we have 10,000 models


class ModelCreator:
    def __init__(self, model_params: ModelParams = None, random_init=False):
        if not model_params and not random:
            raise ValueError("Random must be set to False if model_params are not initialized")
        self.model_params = model_params
        self.random_init = random_init

    def create(self):

        if self.random_init:
            self.model_params = ModelParams.random_init()

        num_features = AffinityDB().get_num_features()
        layers = []

        for i in range(self.model_params.layer_count):
            layers.extend([tf.keras.layers.Dense(self.model_params.layer_sizes[i],
                                                 kernel_regularizer=l1(self.model_params.reg_rate)),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.PReLU()
                           ])

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_features,)),
            *layers,
            tf.keras.layers.Dense(1)
        ])

        try:
            optimizer = getattr(tf.keras.optimizers,
                                self.model_params.algo)(learning_rate=self.model_params.lr)
        except ValueError:
            raise ValueError("Unknown optimization algorithm")

        model.compile(optimizer=optimizer,
                      loss=self.model_params.loss,
                      metrics=[self.model_params.loss])

        return Model(model)


class UserScoresPredictor():

    def __init__(self, user_name, model, site="MAL", shows_to_take="all"):
        self.user_name = user_name
        self.model = model
        user_db = UserDB()
        self.user_row = user_db.get_user_db_entry(user_name, site)
        self.tags = Tags()
        self.user_db = UserDB()
        self.aff_db = AffinityDB()
        self.anime_db = AnimeDB(anime_database_updated_name)
        try:
            self.data = load_pickled_file(data_path / "general_data.pickle")
        except FileNotFoundError:
            self.data = GeneralData().generate_data()

        user_scores_dict = self.user_row.row(0, named=True)
        scored_amount = user_scores_dict["Scored Shows"]
        [user_scores_dict.pop(x, 0) for x in UserDB.stats]

        self.user = User(user_name, scores=user_scores_dict,
            scored_amount=scored_amount)

        self.shows_to_take = shows_to_take
        self.user_shows_df_with_name = self.get_user_db(self.user, self.shows_to_take)

    def predict_scores(self):
        least_fav_tags = [Tags.format_doubletag(key) for key, value in sorted(
            self.user.tag_affinity_dict.items(), key=lambda x: (
                    x[1] + 2 * self.user.tag_pos_affinity_dict[x[0]]))[0:20]]

        fav_tags = [Tags.format_doubletag(key) for key, value in sorted(
            self.user.tag_affinity_dict.items(), reverse=True, key=lambda x: (
                    x[1] + 2 * self.user.tag_pos_affinity_dict[x[0]]))[0:20]]

        predictions, predictions_no_watched = self.fetch_predictions()

        predictions_sorted_by_diff = sorted(predictions,
                                            reverse=True,
                                            key=lambda x:
                                            (x['PredictedScore'] - x['MALScore']))

        return (predictions[0:400], predictions_sorted_by_diff[0:400],
                fav_tags, least_fav_tags)

    def get_user_db(self, user, shows_to_take, site="MAL"):
        # move to affinitydb in future version
        aff_db_entry_creator = AffDBEntryCreator(user, self.data, self.tags, for_predict=True)
        aff_db_entry_creator.create_db_entries_from_user_list(
            shuffle_list=False, shows_to_take=shows_to_take,
            db_type=1, site=site)

        user_db_entries = aff_db_entry_creator.aff_db_entry_dict
        user_shows_df_with_name = pd.DataFrame(user_db_entries).fillna(0)
        return user_shows_df_with_name

    def fetch_predictions(self):

        def format_predictions(preds, show_names):

            rounded_preds = [round(x, 1) for x in preds]
            pred_dict = {x[0]: x[1] for x in sorted(list(zip(show_names,rounded_preds)))}
            return pred_dict

        def calculate_prediction_adjustment(main_entry_predicted_score, main_entry, max_score_entry,
                                            show_predicted_score, main_entry_mean_score, max_score_entry_mean_score):
            """
            Mitigates wrongfully high predictions that result from taking a show's final predicted score
            as the maximum of all entries.

            A show's predicted score will be lowered if the show meets ALL of the following conditions :

            1 - The predicted score for either the main or the highest scored (on MAL) entries of the show
            is lower than the MAL scores for these entries (For example if Gintama S1 prediction = 6.3 it meets
            the condition because the MAL score is 8.9, same if the highest scored Gintama season prediction was 6.3)

            2 - The prediction for the main entry of the show is lower relative to the overall prediction for that
            show. (For example Gintama S1 prediction = 6.3, Gintama S8 prediction = 9.4)

            3 - The predicted score for the main entry is no more than 1 point above the user's mean score.

            4 - The predicted score for the entire show (max of all entries) is higher than the user's mean.
            The logic behind the function is to lower the predictions of any show whose
            main entry was given a significantly lower rating compared to a later entry
            (which is ignored if we go by the general formula, which is simply to take
            the highest prediction among all entries).

            The overall idea is simply that if there is a large discrepancy between the prediction for the
            main entry of a show (Gintama) and another season/entry that might have different tags
            (the more serious arcs of Gintama), the final predicted show should take the main entry into account more
            than those more serious arcs (unlike normally, where if quality and type

            This is by no means a perfect system, but it vastly improved the model's original predictions,
            and is much better than taking the average of all entries (due to outliers and things like OVAs).

            """

            m1 = min(main_entry_predicted_score - main_entry_mean_score,
                     show_predicted_score - max_score_entry_mean_score)
            m2 = main_entry_predicted_score - show_predicted_score
            m3 = main_entry_predicted_score - self.user.mean_of_watched - 1
            m4 = self.user.mean_of_watched - show_predicted_score
            return max(min(0, m1), min(0, m2), min(0, m3), min(0, m4))  # We never want to increase the score hence min(0,m1)

        def process_predictions():

            try:
                max_pos_affs_per_entry = {
                self.user_shows_df_with_name['Show Name'][i]:
                    (self.user_shows_df_with_name['Single Max Pos Affinity'][i]
                     + self.user_shows_df_with_name['Double Max Pos Affinity'][i])
                for i in range(len(self.user_shows_df_with_name))}
            except TypeError:
                max_pos_affs_per_entry = {
                self.user_shows_df_with_name['Show Name'][i]: 0
                for i in range(len(self.user_shows_df_with_name))}

            predictions_per_show = {}
            max_pos_affs_per_show = {}
            processed_entries = []

            for entry in self.user.entry_list:
                if entry in processed_entries:
                    continue
                try:
                    main_show = self.tags.entry_tags_dict_nls_updated[entry]['Main']
                    main_score = predictions[main_show]
                    max_score = main_score
                    max_score_entry = entry
                    max_pos_aff = max_pos_affs_per_entry[main_show]
                except KeyError:
                    continue
                for entry, length_coeff in self.tags.show_tags_dict_nls_updated[main_show]['Related'].items():
                    if length_coeff == 1 and entry in self.user.entry_list and predictions[entry] > max_score:
                        max_score = predictions[entry]
                        max_score_entry = entry
                        max_pos_aff = max_pos_affs_per_entry[entry]
                    processed_entries.append(entry)
                predictions_per_show[main_show] = max_score + calculate_prediction_adjustment(main_score,
                                                  main_show, max_score_entry,
                                                  max_score, mean_scores[main_show], mean_scores[max_score_entry])

                max_pos_affs_per_show[main_show] = round(max_pos_aff, 3)
            return predictions_per_show, max_pos_affs_per_show

        user_shows_df = self.user_shows_df_with_name.drop('Show Name', axis=1)
        normalized_df = AffinityDB.normalize_aff_df(data=self.data,
                                                    df=user_shows_df, for_predict=True)
        normalized_df = normalized_df.fillna(0)
        normalized_df = AffinityDB.filter_df_for_model(normalized_df)

        mean_scores = self.anime_db.mean_scores

        if not self.shows_to_take.startswith("watched"):
            features = normalized_df.values
        else:
            features = normalized_df.iloc[:, :-1].values

        predictions = self.model.model.predict(features)
        predictions = [min(10, float(x[0])) for x in predictions]

        names = self.user_shows_df_with_name['Show Name'].to_list()

        predictions = format_predictions(predictions, names)

        predictions_per_show, max_pos_affs_per_show = process_predictions()

        mean_scores_of_predictions = [self.tags.get_max_score_of_show(
            show, scores=mean_scores) for show in predictions_per_show.keys()]

        user_watched_entries = self.user_row.select(self.user.entry_list)
        user_watched_entries = user_watched_entries.to_dict(as_series=False)
        user_watched_entries = {entry: score[0] for entry, score in user_watched_entries.items()}

        user_scores_per_show = [self.tags.get_max_score_of_show(show, scores=user_watched_entries)
                                          for show in predictions_per_show.keys()]

        years_dict = self.anime_db.years
        seasons_dict = self.anime_db.converted_seasons

        prediction_data = sorted(list(zip(predictions_per_show.items(), mean_scores_of_predictions,
                                          user_scores_per_show, max_pos_affs_per_show.values())),
                                 reverse=True, key=lambda x: (x[0][1], x[3]))

        new_predictions_list = []
        for pred in prediction_data:
            predictions_per_show = {}
            show_name = pred[0][0]
            predictions_per_show['ShowName'] = show_name
            predictions_per_show['PredictedScore'] = pred[0][1]
            predictions_per_show['UserScore'] = pred[2]
            predictions_per_show['MALScore'] = pred[1]
            predictions_per_show['Year'] = int(years_dict[show_name])
            predictions_per_show['Season'] = seasons_dict[show_name]
            new_predictions_list.append(predictions_per_show)

        new_predictions_list = sorted(new_predictions_list, key=lambda x: x['PredictedScore'], reverse=True)
        new_predictions_list_no_watched = [x for x in new_predictions_list if not x['UserScore']]
        return new_predictions_list, new_predictions_list_no_watched


class ModelTester:
    def __init__(self):
        self.user_db = UserDB()
        try:
            self.data = load_pickled_file(data_path / "general_data.pickle")
        except FileNotFoundError:
            print("General data file is required for normalization. Unable to test models.")

    @staticmethod
    def _write_model_details(model_num):

        model = Model(tf.keras.models.load_model(
            models_path / f"Main_prediction_model{model_num if model_num != 1 else ''}.h5"))

        with open(data_path / "ModelTests" / f"ModelTest{model_num}.txt", "w") as f:
            f.write(f"Optimizer : {model.model.optimizer._name}\n")
            f.write(f"Loss : {model.model.loss.__name__}\n")
            f.write("Layers :\n")
            for layer in model.model.layers:
                try:
                    f.write(f"{str(layer.units)}\n")
                except AttributeError:
                    continue

    def test_all(self, starting_model_index=1):

        shows_to_take = "all"
        with open(data_path / "ModelTests" / "ModelTestUsernames.txt", 'r') as f:
            test_usernames = [username.strip() for username in f.readlines()]

        i = starting_model_index
        while True:
            try:
                print(f"Writing details of model {i}")
                self._write_model_details(i)
                i += 1
            except (OSError, FileNotFoundError):
                break

        for username in test_usernames:

            i = starting_model_index
            starting_model = Model(tf.keras.models.load_model(models_path / f"Main_prediction_model{i}.h5"))
            predictor = UserScoresPredictor(user_name=username, model=starting_model, site="MAL",
                                            shows_to_take=shows_to_take)
            while True:
                with open(data_path / "ModelTests" / f"ModelTest{i}.txt", "a", encoding='utf-8') as f:
                    if i > starting_model_index:
                        model_filename = models_path / f"Main_prediction_model{i if i != 1 else ''}.h5"
                        try:
                            model = Model(tf.keras.models.load_model(model_filename))  # remove the models_path?
                        except (OSError, FileNotFoundError):
                            break
                        predictor.model = model
                    print(f"Fetching predictions of user {username} for model {i}")
                    predictions, predictions_no_watched = predictor.fetch_predictions()
                    # errors = self.calculate_error(predictions, user.mean_of_watched)
                    print(f"Writing predictions of user {username} for model {i}")

                    f.write(f"\n\n{username}\n")
                    # f.write(f"Errors : \n{errors}\n")
                    # f.write(errors)
                    f.write("Predictions\n")

                    for prediction in predictions[0:50]:
                        f.write(f"{prediction}\n")

                    # predictions = self.adjust_predictions(predictions)
                    predictions = sorted(predictions, reverse=True,
                                         key=lambda x: x['PredictedScore'] - x['MALScore'])
                    f.write("-------------------------------------------------------")

                    for prediction in predictions[0:50]:
                        f.write(f"{prediction}\n")

                i += 1

    @staticmethod
    def calculate_error(prediction_list, user_mean_score):
        errors = np.zeros(10)
        error_counts = np.zeros(10)

        pred_average = np.average([x['PredictedScore'] for x in prediction_list])
        pred_diff = user_mean_score - pred_average

        for pred_dict in prediction_list:
            if pred_dict['UserScore']:
                error_counts[pred_dict['UserScore'] - 1] += 1
                if pred_dict['UserScore'] > pred_dict['PredictedScore']:
                    errors[pred_dict['UserScore'] - 1] += abs(pred_dict['UserScore'] - pred_dict['PredictedScore'])

        for i in range(10):
            try:
                errors[i] /= error_counts[i]
                errors[i] -= pred_diff
            except ZeroDivisionError:
                errors[i] = 0
        return errors


def add_image_urls_to_predictions(recs):
    show_names = [x['ShowName'] for x in recs[0:400]]

    # Fetch all AnimeDataUpdated records at once
    anime_data = AnimeDataUpdated.objects.filter(name__in=show_names)
    anime_data_dict = {anime.name: anime.image_url for anime in anime_data}

    # Initialize the list of image URLs with error handling
    image_urls = []
    for rec in recs[0:400]:
        try:
            image_url = anime_data_dict.get(rec['ShowName'])
            if image_url is None:
                raise ValueError("Show Not Found")

            image_url_split = image_url.split("/")
            suffix1 = int(image_url_split[-2])
            suffix2 = int(image_url_split[-1].split(".")[0])
            image_urls.append((suffix1, suffix2))
        except Exception as ex:
            # Log the error and append (0, 0) as fallback
            recs_logger.error(f"Failed to process URL for show {rec['ShowName']}. {ex}")
            image_urls.append((0, 0))

    # Now merge the image URL suffixes into the records
    recs = [
        rec | {'ImageUrlSuffix1': image_urls[i][0], 'ImageUrlSuffix2': image_urls[i][1]}
        if i < 400 else rec
        for i, rec in enumerate(recs)
    ]
    return recs



