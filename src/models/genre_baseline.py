from models.base_recommender import BaseRecommender

from collections import defaultdict
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class GenreBaselineModel(BaseRecommender):
    def __init__(self, anime_data):
        """
        Parameters:
        - anime_data: an instance of AnimeData class or similar, which has a DataFrame with 'Genres' and 'MAL_ID'.
                      It must provide a way to get genres by anime_id and a genre_mean_score function.
        """
        super().__init__()
        self.anime_data = anime_data
        self.user_genre_means = None
        self.global_genre_mean = None

    def fit(self, train_df):
        """
        Precompute:
        - user_genre_means: {user_id: {genre: avg_rating_for_that_genre}}
        - global_genre_mean: fallback mean rating if user never rated any anime with that genre.
        
        train_df: A DataFrame that must contain at least columns: user_id, anime_id, rating
        """

        # Extract unique genres from anime_data if needed.
        # We'll build user_genre_means by going through train data.
        
        # 1. Build a mapping of anime_id to genres
        anime_genres_map = {}
        for idx, row in self.anime_data.df.iterrows():
            mal_id = row['MAL_ID']
            genres = row['Genres'] if isinstance(row['Genres'], list) else []
            anime_genres_map[mal_id] = genres

        # 2. For each user-anime-rating in train_df, record the user's rating by genre
        # We'll create a structure: user_ratings_by_genre[user_id][genre].append(rating)
        user_ratings_by_genre = defaultdict(lambda: defaultdict(list))

        for _, entry in train_df.iterrows():
            u = entry['user_id']
            i = entry['anime_id']
            r = entry['rating']
            # If rating is NaN or something invalid, skip
            if pd.isna(r):
                continue
            anime_genres = anime_genres_map.get(i, [])
            for g in anime_genres:
                user_ratings_by_genre[u][g].append(r)

        # 3. Compute means for each user-genre
        self.user_genre_means = {}
        for u, genre_dict in user_ratings_by_genre.items():
            self.user_genre_means[u] = {}
            for g, ratings in genre_dict.items():
                if len(ratings) > 0:
                    self.user_genre_means[u][g] = np.mean(ratings)
                else:
                    self.user_genre_means[u][g] = np.nan

        # 4. Compute a global fallback genre mean
        # If you want a global mean for all anime regardless of genre:
        #   global_mean = train_df['rating'].mean()
        # But the request: "go back to the global genre mean we just implement"
        # If we interpret "global genre mean" as using the `genre_mean_score`
        # function, we need a fallback genre. The instructions are a bit ambiguous:
        # "for the genres in that anime, ... fallback to the global genre mean"
        # implies we compute mean score for these genres from AnimeData.
        # We'll store AnimeData so we can compute on-the-fly in forward().
        # To speed things up, we can compute a pre-mean for each genre individually.

        # Precompute global means for each genre so we don't have to do it at runtime
        # We'll first gather all unique genres
        all_genres = set()
        for g_list in anime_genres_map.values():
            all_genres.update(g_list)
        self.global_genre_means = {}
        for g in all_genres:
            gm = self.anime_data.genre_mean_score([g])
            self.global_genre_means[g] = gm if gm is not None else np.nan

    def forward(self, user_ids, item_ids):
        """
        Predict rating for each user_id, item_id pair.
        Steps:
        - For the given item_id, find the genres.
        - Check if user has rated any anime from those genres.
          If yes, use mean of user's genre means for those genres.
          If no, fallback to mean of global genre means for those genres.
        - If global genre means are also NaN (shouldn't happen if we have data), just pick a global average rating.
        
        user_ids, item_ids: torch tensors of the same shape
        """
        # Convert to numpy for easy iteration
        user_ids_np = user_ids.detach().cpu().tolist()
        item_ids_np = item_ids.detach().cpu().tolist()

        predictions = []
        for u, i in zip(user_ids_np, item_ids_np):
            # Get genres
            row = self.anime_data.df[self.anime_data.df['MAL_ID'] == i]
            if row.empty:
                # If no info about this anime, fallback to a global average
                # For simplicity, use mean of all global_genre_means that are not NaN
                global_values = [v for v in self.global_genre_means.values() if not pd.isna(v)]
                pred = np.mean(global_values) if global_values else 5.0
                predictions.append(pred)
                continue

            genres = row.iloc[0]['Genres']
            if not isinstance(genres, list):
                genres = []

            # Check user's genre means
            user_mean_dict = self.user_genre_means.get(u, {})
            user_genre_ratings = [user_mean_dict[g] for g in genres if g in user_mean_dict and not pd.isna(user_mean_dict[g])]

            if len(user_genre_ratings) > 0:
                # Use user's average for these genres
                pred = np.mean(user_genre_ratings)
            else:
                # Fallback to global genre means for these genres
                global_genre_ratings = [self.global_genre_means[g] for g in genres if g in self.global_genre_means and not pd.isna(self.global_genre_means[g])]
                if len(global_genre_ratings) == 0:
                    # Fallback to an overall mean if no global genre mean is available
                    global_values = [v for v in self.global_genre_means.values() if not pd.isna(v)]
                    pred = np.mean(global_values) if global_values else 5.0
                else:
                    pred = np.mean(global_genre_ratings)

            predictions.append(pred)

        return torch.tensor(predictions, dtype=torch.float)


# Example usage (assuming you have train_df and anime_data already):
# anime_data = AnimeData(filepath='data/raw/anime.csv')
# model = GenreBaselineModel(anime_data)
# model.fit(train_df)  # train_df must have user_id, anime_id, rating
# user_ids = torch.tensor([1, 2, 3])
# item_ids = torch.tensor([20, 30, 40])
# preds = model(user_ids, item_ids)
# print(preds)