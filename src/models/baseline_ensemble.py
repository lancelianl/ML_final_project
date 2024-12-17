import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

interaction_matrix = pd.read_csv("/Users/xianghuang/Desktop/ml/final_dataset/interaction_matrix.csv", index_col=0)
anime_data = pd.read_csv("/Users/xianghuang/Desktop/ml/final_dataset/anime.csv")

anime_data["MAL_ID"] = anime_data["MAL_ID"].astype(str)
interaction_matrix.columns = interaction_matrix.columns.astype(str)

column_popularity = (interaction_matrix != 0).sum(axis=0)
anime_data["popularity"] = anime_data["MAL_ID"].map(column_popularity).fillna(0).astype(int)


# Baseline Model
# Popularity-Based Recommendation
def popularity_based_recommender(n=10):
    """
    Recommend the most popular anime based on the number of ratings.
    """
    popular_anime = anime_data.sort_values(by="popularity", ascending=False)
    return popular_anime.head(n)[["MAL_ID", "Name", "popularity"]]

# Genre-Based Recommendation
def genre_based_recommender(user_id, n=10):
    """
    Recommend anime from genres the user has rated highly.
    """
    anime_data["MAL_ID"] = anime_data["MAL_ID"].astype(str)
    interaction_matrix.index = interaction_matrix.index.astype(str)

    user_ratings = interaction_matrix.loc[str(user_id)].dropna()
    if user_ratings.empty:
        return pd.DataFrame(columns=["MAL_ID", "Name", "Genres"])

    favorite_anime_ids = user_ratings[user_ratings >= user_ratings.mean()].index
    anime_data["Genres"] = anime_data["Genres"].apply(lambda x: x.split(",") if isinstance(x, str) else [])

    favorite_genres = anime_data[anime_data["MAL_ID"].isin(favorite_anime_ids)]["Genres"]
    favorite_genres = favorite_genres.explode().value_counts().index[:3]  # Top 3 genres

    anime_with_genres = anime_data[
        anime_data["Genres"].apply(lambda x: any(genre in x for genre in favorite_genres))
    ]
    return anime_with_genres.sort_values(by="popularity", ascending=False).head(n)[["MAL_ID", "Name", "Genres"]]


# Ensemble Method
class EnsembleRecommender:
    def __init__(self, anime_data):
        self.anime_data = anime_data
           
    def recommend(self, user_id, collab_predictions, content_predictions, n=10):
        """
        Combines collaborative filtering and content-based filtering predictions.
        """
        ensemble_scores = 0.6 * collab_predictions + 0.4 * content_predictions

        self.anime_data["ensemble_score"] = ensemble_scores
        top_recommendations = self.anime_data.sort_values(by="ensemble_score", ascending=False).head(n)

        return top_recommendations[["MAL_ID", "Name", "ensemble_score"]]


# np.random.seed(42)

# Baseline recommendations
# print("Popularity-Based Recommendations:")
# print(popularity_based_recommender(n=5))

# print("Genre-Based Recommendations:")
# print(genre_based_recommender(user_id=1, n=5))

# Generate reproducible random predictions
#collab_predictions = np.random.rand(len(anime_data))
#content_predictions = np.random.rand(len(anime_data))

# Ensemble recommendations
# ensemble_recommender = EnsembleRecommender(anime_data)
# print("Ensemble Recommendations:")
# print(ensemble_recommender.recommend(user_id=1, collab_predictions=collab_predictions, content_predictions=content_predictions, n=5))

