import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class RatingDataset(Dataset):
    def __init__(self, dataframe, include_watching_status=False, include_watched_episodes=False):
        """
        PyTorch dataset for user-anime-rating data.
        """
        self.data = dataframe.reset_index(drop=True)
        self.include_watching_status = include_watching_status
        self.include_watched_episodes = include_watched_episodes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_id = row['user_id']
        anime_id = row['anime_id']
        rating = row['rating']

        if self.include_watching_status:
            watching_status = row['watching_status']
        if self.include_watched_episodes:
            watched_episodes = row['watched_episodes']

        # Convert to tensors
        user_id_t = torch.tensor(user_id, dtype=torch.long)
        anime_id_t = torch.tensor(anime_id, dtype=torch.long)
        rating_t = torch.tensor(rating, dtype=torch.float)

        # Conditionally return extra features
        if self.include_watching_status and self.include_watched_episodes:
            return user_id_t, anime_id_t, rating_t, torch.tensor(watching_status, dtype=torch.long), torch.tensor(watched_episodes, dtype=torch.long)
        elif self.include_watching_status:
            return user_id_t, anime_id_t, rating_t, torch.tensor(watching_status, dtype=torch.long)
        elif self.include_watched_episodes:
            return user_id_t, anime_id_t, rating_t, torch.tensor(watched_episodes, dtype=torch.long)
        else:
            return user_id_t, anime_id_t, rating_t

def preprocess_data(
    filepath='../../data/raw/animelist.csv',
    user_range=None,
    include_watching_status=False,
    include_watched_episodes=False,
    zero_strategy='none',
    random_state=42,
    save_interaction_matrix=True
):
    """
    Preprocess the animelist data and return train, val, test datasets with an exact 0.8/0.1/0.1 split.

    Parameters:
    - filepath: path to the animelist.csv
    - user_range: int or None, if int, includes only first `user_range` unique users.
    - include_watching_status: bool
    - include_watched_episodes: bool
    - zero_strategy: str in {'average', 'discard', 'none'}
    - random_state: int for reproducibility
    - save_interaction_matrix: bool, whether to save processed interaction matrix for the train set

    Returns:
    - train_dataset, val_dataset, test_dataset: RatingDataset objects
    """

    # Load data
    df = pd.read_csv(filepath)
    # Ensure data types
    df['user_id'] = df['user_id'].astype(int)
    df['anime_id'] = df['anime_id'].astype(int)
    df['rating'] = df['rating'].astype(int)

    # Apply user_range if given
    if user_range is not None:
        unique_users = df['user_id'].unique()
        if user_range < len(unique_users):
            allowed_users = unique_users[:user_range]
            df = df[df['user_id'].isin(allowed_users)]

    # Handle zero_strategy
    if zero_strategy == 'discard':
        # Remove rows where rating == 0
        df = df[df['rating'] != 0].copy()
    elif zero_strategy == 'average':
        user_means = df[df['rating'] != 0].groupby('user_id')['rating'].mean()
        global_mean = df[df['rating'] != 0]['rating'].mean() if df[df['rating'] != 0].shape[0] > 0 else 0
        def replace_zero(row):
            if row['rating'] == 0:
                user_id = row['user_id']
                return user_means[user_id] if user_id in user_means else global_mean
            else:
                return row['rating']
        df['rating'] = df.apply(replace_zero, axis=1)
    elif zero_strategy == 'none':
        pass
    else:
        raise ValueError("zero_strategy must be one of ['average', 'discard', 'none']")

    # Shuffle the DataFrame
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    df = df.iloc[shuffled_indices].reset_index(drop=True)

    # Compute split sizes
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)  # after train_end

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Create the datasets
    train_dataset = RatingDataset(train_df, 
                                 include_watching_status=include_watching_status,
                                 include_watched_episodes=include_watched_episodes)
    val_dataset = RatingDataset(val_df, 
                                include_watching_status=include_watching_status,
                                include_watched_episodes=include_watched_episodes)
    test_dataset = RatingDataset(test_df, 
                                 include_watching_status=include_watching_status,
                                 include_watched_episodes=include_watched_episodes)

    # Save interaction matrix for training data if desired
    if save_interaction_matrix:
        interaction_matrix = train_df.pivot(index='user_id', columns='anime_id', values='rating').fillna(0)
        os.makedirs('data/processed', exist_ok=True)
        interaction_matrix.to_csv('../data/processed/interaction_matrix.csv')

    return train_dataset, val_dataset, test_dataset

# Example usage:
if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = preprocess_data(
        filepath='data/raw/animelist.csv',
        user_range=10000,
        include_watching_status=True,
        include_watched_episodes=True,
        zero_strategy='average',
        random_state=42,
        save_interaction_matrix=True
    )

    # Create DataLoaders if needed
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch in train_loader:
        # Just to check first batch
        break
