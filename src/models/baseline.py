import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from models.base_recommender import BaseRecommender

class BaselineModel(BaseRecommender):
    def __init__(self, anime_data):
        """
        Predicts ratings based on user averages and computes MSE.
        """
        super().__init__()
        self.anime_data = anime_data
        self.user_mean_ratings = None
        self.global_mean_rating = None

    def fit(self, train_df):
        """
        Train the baseline model by computing:
        - Mean rating for each user (user-level average)
        - Global mean rating as a fallback
        """
        self.user_mean_ratings = train_df.groupby('user_id')['rating'].mean().to_dict()
        self.global_mean_rating = train_df['rating'].mean()
        print(f"Global Mean Rating: {self.global_mean_rating:.4f}")

    def forward(self, user_ids, item_ids):
        """
        Predict ratings for user-item pairs based on user mean ratings or global mean.
        """
        predictions = []
        for user_id in user_ids:
            pred = self.user_mean_ratings.get(user_id, self.global_mean_rating)
            predictions.append(pred)
        return torch.tensor(predictions, dtype=torch.float32)

    def evaluate_mse(self, test_df):
        """
        Evaluate the model by predicting ratings and computing the MSE.
        """
        user_ids = test_df['user_id'].tolist()
        predictions = self.forward(user_ids, test_df['anime_id'].tolist())
        actual_ratings = torch.tensor(test_df['rating'].tolist(), dtype=torch.float32)
        mse = nn.MSELoss()(predictions, actual_ratings).item()
        return mse

