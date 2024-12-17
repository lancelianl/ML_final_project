import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.base_recommender import BaseRecommender


class PopularityBaselineModel(BaseRecommender):
    """
    Popularity-based recommender system. Recommends the most popular anime 
    based on the number of non-zero ratings.
    """
    def __init__(self, anime_data, interaction_matrix):
        """
        Parameters:
        - anime_data: DataFrame containing anime metadata.
        - interaction_matrix: DataFrame of user-item interaction ratings.
        """
        super().__init__()
        self.anime_data = anime_data
        self.interaction_matrix = interaction_matrix
        self.popularity_scores = None

    def fit(self):
        """
        Precompute popularity scores for all anime based on non-zero ratings.
        """
        column_popularity = (self.interaction_matrix != 0).sum(axis=0)
        self.anime_data["popularity"] = self.anime_data["MAL_ID"].map(column_popularity).fillna(0).astype(int)
        self.popularity_scores = self.anime_data.sort_values(by="popularity", ascending=False)

    def recommend(self, n=10):
        """
        Recommend the top-n most popular anime.
        """
        return self.popularity_scores.head(n)[["MAL_ID", "Name", "popularity"]]


class GenreBaselineModel(BaseRecommender):
    """
    Genre-based recommender system. Recommends anime from genres that users 
    have rated highly.
    """
    def __init__(self, anime_data, interaction_matrix):
        """
        Parameters:
        - anime_data: DataFrame containing anime metadata.
        - interaction_matrix: DataFrame of user-item interaction ratings.
        """
        super().__init__()
        self.anime_data = anime_data
        self.interaction_matrix = interaction_matrix

    def recommend(self, user_id, n=10):
        """
        Recommend anime based on the user's favorite genres.

        Parameters:
        - user_id: ID of the user to generate recommendations for.
        - n: Number of recommendations to return.
        """
        self.anime_data["MAL_ID"] = self.anime_data["MAL_ID"].astype(str)
        self.interaction_matrix.index = self.interaction_matrix.index.astype(str)
        user_ratings = self.interaction_matrix.loc[str(user_id)].dropna()
        if user_ratings.empty:
            return pd.DataFrame(columns=["MAL_ID", "Name", "Genres"])

        favorite_anime_ids = user_ratings[user_ratings >= user_ratings.mean()].index
        self.anime_data["Genres"] = self.anime_data["Genres"].apply(lambda x: x.split(",") if isinstance(x, str) else [])

        favorite_genres = self.anime_data[self.anime_data["MAL_ID"].isin(favorite_anime_ids)]["Genres"]
        favorite_genres = favorite_genres.explode().value_counts().index[:3]  # Top 3 genres

        anime_with_genres = self.anime_data[
            self.anime_data["Genres"].apply(lambda x: any(genre in x for genre in favorite_genres))
        ]
        return anime_with_genres.sort_values(by="popularity", ascending=False).head(n)[["MAL_ID", "Name", "Genres"]]


class EnsembleRecommender(BaseRecommender):
    """
    Ensemble recommender system that combines collaborative filtering and content-based predictions.
    """
    def __init__(self, anime_data):
        """
        Parameters:
        - anime_data: DataFrame containing anime metadata.
        """
        super().__init__()
        self.anime_data = anime_data

    def recommend(self, user_id, collab_predictions, content_predictions, n=10):
        """
        Combine collaborative filtering and content-based predictions using a weighted average.

        Parameters:
        - user_id: User ID to generate recommendations for.
        - collab_predictions: Collaborative filtering prediction scores.
        - content_predictions: Content-based prediction scores.
        - n: Number of recommendations to return.
        """
        ensemble_scores = 0.6 * collab_predictions + 0.4 * content_predictions
        self.anime_data["ensemble_score"] = ensemble_scores
        top_recommendations = self.anime_data.sort_values(by="ensemble_score", ascending=False).head(n)

        return top_recommendations[["MAL_ID", "Name", "ensemble_score"]]
