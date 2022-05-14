import abc
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime



class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.initialize_predictor(ratings)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """

        true_ratings['predict'] = true_ratings.apply(
            lambda sample: self.predict(int(sample[0]), int(sample[1]), int(sample[3]))
            , axis=1)
        return np.sqrt(((true_ratings['predict'] - true_ratings['rating']) ** 2).mean())


class BaselineRecommender(Recommender):

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.r_avg = ratings["rating"].mean()
        self.bu = []
        self.bi = []
        all_users = set(ratings['user'].sort_index())
        for user in all_users:
            subset_df = ratings[ratings["user"] == user]
            self.bu.append(subset_df['rating'].mean() - self.r_avg)
        all_items = set(ratings['item'].sort_index())
        for item in all_items:
            subset_df = ratings[ratings["item"] == item]
            self.bi.append(subset_df['rating'].mean() - self.r_avg)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        curr_prediction = self.r_avg + self.bu[int(user)] + self.bi[int(item)]
        if curr_prediction < 0.5:
            return 0.5
        elif curr_prediction > 5:
            return 5
        else:
            return float(curr_prediction)


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.r_avg = ratings["rating"].mean()
        self.bu = []
        self.bi = []
        self.r_new = ratings.copy()
        self.r_new['r_wave'] = self.r_new.apply(lambda row: row[2] - self.r_avg, axis=1)
        self.baserec = BaselineRecommender(ratings)
        self.all_users = set(ratings['user'].sort_index())
        for user in self.all_users:
            subset_df = ratings[ratings["user"] == user]
            self.bu.append(subset_df['rating'].mean() - self.r_avg)
        self.all_items = set(ratings['item'].sort_index())
        for item in self.all_items:
            subset_df = ratings[ratings["item"] == item]
            self.bi.append(subset_df['rating'].mean() - self.r_avg)
        self.ratings_per_user_item = self.r_new.pivot(index='user', columns='item', values='r_wave').to_numpy()
        self.ratings_per_user_item_zeros = np.nan_to_num(self.ratings_per_user_item)
        self.similarity = cosine_similarity(self.ratings_per_user_item_zeros)
        self.similarity_absvalues = np.absolute(self.similarity).copy()

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        curr_prediction = self.baserec.predict(user, item, timestamp)
        k_closest_neighboors = self.compute_3_nearest(user, item)
        not_abs = 0
        abs = 0
        r_new_only_item = self.r_new[self.r_new["item"] == item].copy()
        for row, values in k_closest_neighboors:
            abs = abs + values[1]
            neih = values[0]
            r_new_only_item1 = (r_new_only_item[r_new_only_item["user"] == neih])
            r_wave_neih = list(r_new_only_item1['r_wave'])
            if len(r_wave_neih) == 0:
                r_wave_neih1 = 0
            else:
                r_wave_neih1 = r_wave_neih[0]
            not_abs_for_curr_neih = values[2]
            not_abs = not_abs + (not_abs_for_curr_neih * r_wave_neih1)

        if abs != 0:
            curr_prediction = curr_prediction + (not_abs / abs)
        if curr_prediction < 0.5:
            return 0.5
        elif curr_prediction > 5:
            return 5
        else:
            return curr_prediction

    def compute_3_nearest(self, user: int, item: int):
        A = self.r_new[self.r_new["item"] == item]
        users = A["user"].astype(int)
        user_and_similarity = []
        for neih in users:
            user_and_similarity.append((neih, self.similarity_absvalues[int(user), int(neih)],
                                        self.similarity[int(user), int(neih)]))
        B = sorted(enumerate(user_and_similarity), key=lambda i: i[1], reverse=True)

        my_closest = B[:3]
        return my_closest

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """

        subset_df_user1 = self.r_new[self.r_new["user"] == user1].copy()
        subset_df_user2 = self.r_new[self.r_new["user"] == user2].copy()

        inner_joined_total = subset_df_user1.join(subset_df_user2.set_index(["item"]),
                                                  lsuffix="_first", rsuffix="_second", on=["item"]).copy()
        inner_joined_total.dropna(axis=0, inplace=True)
        if inner_joined_total.empty:
            return 0
        dot = inner_joined_total.apply(lambda row: row[4] * row[8], axis=1).sum()
        sum1 = (subset_df_user1['r_wave'] ** 2).sum()
        sum2 = (subset_df_user2['r_wave'] ** 2).sum()

        return dot / ((sum1 * sum2) ** 0.5)


class LSRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.r_avg = ratings["rating"].mean()
        self.r_new = ratings.copy()
        self.r_new['r_wave'] = self.r_new.apply(lambda row: row[2] - self.r_avg, axis=1)
        self.num_users = len((ratings['user']).unique())
        self.num_items = len((ratings['item']).unique())
        self.matrix_X = np.zeros(shape=(ratings.shape[0], self.num_items + self.num_users + 3))
        for i, row in self.r_new.iterrows():
            temp_user = int(row[0])
            temp_item = int(row[1])
            self.matrix_X[i, temp_user] = 1
            self.matrix_X[i, temp_item + self.num_users - 1] = 1
            time = pd.datetime.fromtimestamp(row[3])
            if 6 < time.hour < 18:
                self.matrix_X[i, self.num_users + self.num_items] = 1
            else:
                self.matrix_X[i, self.num_users + self.num_items + 1] = 1
            if time.date == 5 or time.date == 4:
                self.matrix_X[i, self.num_users + self.num_items + 2] = 1

        self.y = self.r_new['r_wave']

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        first_add = self.beta[user] + self.beta[self.num_users + item - 1]
        second_add = self.beta[self.num_users + self.num_items] + self.beta[self.num_users + self.num_items + 1] + \
                     self.beta[self.num_users + self.num_items + 2]

        curr_prediction = first_add + second_add + self.r_avg
        if curr_prediction < 0.5:
            return 0.5
        elif curr_prediction > 5:
            return 5
        else:
            return curr_prediction

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        self.beta = np.linalg.lstsq(self.matrix_X, self.y, rcond=None)[0]
        return (self.matrix_X, self.beta, self.y)


class CompetitionRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.r_avg1 = ratings["rating"].mean()
        self.r_new1 = ratings.copy()
        self.r_new1['r_wave'] = self.r_new1.apply(lambda row: row[2] - self.r_avg1, axis=1)
        self.bu1 = []
        self.bi1 = []
        all_users1 = set(ratings['user'].sort_index())
        for user1 in all_users1:
            subset_df1 = ratings[ratings["user"] == user1]
            self.bu1.append(subset_df1['rating'].mean() - self.r_avg1)
        all_items1 = set(ratings['item'].sort_index())
        for item1 in all_items1:
            subset_df2 = ratings[ratings["item"] == item1]
            self.bi1.append(subset_df2['rating'].mean() - self.r_avg1)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """

        ## in order to minimize RMSE, we tried to give more weight to bi entry and less weight to bu.
        ## we tried some combinations and found that 0.65, 0.35 is the best.

        bu_user = self.bu1[int(user)]
        bi_item = self.bi1[int(item)]


        curr_prediction1 = self.r_avg1 + 0.35 * bu_user + 0.65 * bi_item  # 0.95252


        # curr_presiction1 = self.r_avg1 + 0.5 * bu_user + 0.5 * bi_item #0.9565

        # curr_presiction1 = self.r_avg1 + 0.2 * bu_user + 0.8 * bi_item #0.95789

        # curr_presiction1 = self.r_avg1 + 0.05 * bu_user + 0.95 * bi_item #0.9725

        # curr_presiction1 = self.r_avg1 + 0.4 * bu_user + 0.6 * bi_item #0.9528

        # curr_presiction1 = self.r_avg1 + 0.35 * bu_user + 0.65 * bi_item #0.95252

        # curr_presiction1 = self.r_avg1 + 0.3 * bu_user + 0.7 * bi_item #0.95327

        # curr_presiction1 = self.r_avg1 + 0.6 * bu_user + 0.4 * bi_item #0.9644

        if curr_prediction1 < 0.5:
            return 0.5
        elif curr_prediction1 > 5:
            return 5
        else:
            return curr_prediction1
