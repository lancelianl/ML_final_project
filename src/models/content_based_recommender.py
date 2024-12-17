import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import re

from models.base_recommender import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
    def __init__(self, anime_data):
        super().__init__()
        self.anime_data = anime_data
        self.item_features = None
        self.user_profiles = None
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        self.model = None
        self.training_data = None

    def preprocess_items(self):
        df = self.anime_data
        df = df.replace('Unknown', np.nan)

        numeric_cols = ['Score', 'Episodes', 'Members', 'Favorites', 'Watching', 
                        'Completed', 'On-Hold', 'Dropped', 'Plan to Watch']

        # numeric_cols = ['Score']
        one_hot_cols = ['Type', 'Source', 'Rating']
        multi_hot_cols = ['Genres', 'Producers', 'Studios', 'Licensors']

        # Parse 'Duration' column into numerical feature (total minutes)
        def parse_duration(duration):
            if pd.isna(duration):
                return np.nan
            match = re.match(r"(?:(\d+)\s*hr\.)?\s*(?:(\d+)\s*min\.)?", duration)
            if match:
                hours = int(match.group(1)) if match.group(1) else 0
                minutes = int(match.group(2)) if match.group(2) else 0
                return hours * 60 + minutes
            return np.nan

        df['Duration_Minutes'] = df['Duration'].apply(parse_duration)
        numeric_cols.append('Duration_Minutes')  # Add to numeric columns

        # Parse 'Premiered' column into numerical features
        def parse_premiered(premiered):
            if pd.isna(premiered) or premiered == 'Unknown':
                return np.nan, np.nan
            match = re.match(r"(Winter|Spring|Summer|Fall)\s(\d{4})", premiered)
            if match:
                season, year = match.groups()
                season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
                return season_mapping[season], int(year)
            return np.nan, np.nan

        df[['Season', 'Year']] = df['Premiered'].apply(
            lambda x: pd.Series(parse_premiered(x))
        )
        df['Season'].fillna(df['Season'].mode()[0], inplace=True)
        df['Year'].fillna(df['Year'].median(), inplace=True)

        # Combine into a single feature
        df['Premiered_Numeric'] = df['Year'] * 4 + df['Season']
        numeric_cols.append('Premiered_Numeric')  # Add to numeric columns for scaling

        # Fill numeric NaNs and scale
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        numeric_data = StandardScaler().fit_transform(df[numeric_cols])

        # Apply PCA to numeric data
        # pca = PCA(n_components=5)  # Retain 5 principal components
        # pca_data = pca.fit_transform(numeric_data)

        # One-hot encoding
        one_hot_data_list = []
        for col in one_hot_cols:
            one_hot_data = pd.get_dummies(df[col], prefix=col)
            one_hot_data_list.append(one_hot_data)
        one_hot_data = pd.concat(one_hot_data_list, axis=1)

        # Multi-hot encoding
        multi_hot_data_list = []
        for col in multi_hot_cols:
            df[col] = df[col].fillna('')
            vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), binary=True)
            multi_hot_matrix = vectorizer.fit_transform(df[col])
            multi_hot_data_list.append(multi_hot_matrix.toarray())
        multi_hot_data = np.hstack(multi_hot_data_list)

        # Combine features
        self.item_features = np.hstack([numeric_data, one_hot_data.values, multi_hot_data])

        print(len(self.item_features[0]))
        # Map items
        df['item_idx'] = range(len(df))
        self.idx2item = df.set_index('item_idx')['MAL_ID'].to_dict()
        self.item2idx = {v: k for k, v in self.idx2item.items()}

    def build_user_profiles(self, ratings_data):

        unique_users = ratings_data['user_id'].unique() # number of unique users

        # Create mappings
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}

        # Map user_ids and item_ids for building user profiles
        mapped_user_ids = ratings_data['user_id'].map(self.user2idx).values
        mapped_item_ids = ratings_data['anime_id'].map(self.item2idx).values
        ratings = ratings_data['rating'].values

        # Compute weighted features for user profiles
        weighted_features = self.item_features[mapped_item_ids] * ratings[:, None]

        # Aggregate by user
        user_feature_sum = np.zeros((len(unique_users), self.item_features.shape[1]))
        user_rating_sum = np.zeros(len(unique_users))
        np.add.at(user_feature_sum, mapped_user_ids, weighted_features)
        np.add.at(user_rating_sum, mapped_user_ids, ratings)

        user_rating_sum[user_rating_sum == 0] = 1
        self.user_profiles = user_feature_sum / user_rating_sum[:, None]

        # Create a default user profile for unknown users
        self.default_user_profile = np.mean(self.user_profiles, axis=0)

        # Store original IDs in training data
        # This allows forward to do the mapping internally.
        self.training_data = list(zip(ratings_data['user_id'].values,
                                      ratings_data['anime_id'].values,
                                      ratings_data['rating'].values))

    def build_model(self):
        input_dim = self.item_features.shape[1] * 2
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),  # First hidden layer
            nn.ReLU(),
            nn.Linear(64, 32),         # Second hidden layer
            nn.ReLU(),
            nn.Linear(32, 1)           # Output layer (scalar)
        )

    def forward(self, user_ids, item_ids):

        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.tolist()
        if isinstance(item_ids, torch.Tensor):
            item_ids = item_ids.tolist()

        user_indices = []
        for u_id in user_ids:
            if u_id in self.user2idx:
                user_indices.append(self.user2idx[u_id])
            else:
                # Append index for the default profile
                user_indices.append(-1)  # Use -1 to signify unknown user

        item_indices = [self.item2idx[i_id] for i_id in item_ids]

        # Create user feature matrix
        user_vecs = torch.tensor(
            np.vstack([
                self.user_profiles[idx] if idx != -1 else self.default_user_profile
                for idx in user_indices
            ]),
            dtype=torch.float32
        )

        # Retrieve item feature vectors
        item_vecs = torch.tensor(self.item_features[item_indices], dtype=torch.float32)

        x = torch.cat([user_vecs, item_vecs], dim=1)
        return self.model(x).squeeze()

    def fit(self, ratings_data, lr=1e-3, epochs=50, batch_size=64, val_split=0.2, patience=5):
        # Preprocessing and profile building
        self.preprocess_items()
        self.build_user_profiles(ratings_data)
        self.build_model()

        # Prepare DataLoader with original IDs
        train_data = self.training_data
        user_ids = torch.tensor([t[0] for t in train_data], dtype=torch.long)
        item_ids = torch.tensor([t[1] for t in train_data], dtype=torch.long)
        ratings = torch.tensor([t[2] for t in train_data], dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(user_ids, item_ids, ratings)

        # Split dataset into training and validation sets
        num_val = int(val_split * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_val, num_val])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Early stopping parameters
        best_val_loss = float('inf')
        patience_counter = 0

        self.training_losses = []
        self.validation_losses = []
    
        for epoch in range(epochs):
            # Training loop
            self.model.train()
            train_loss = 0
            for u_batch, i_batch, r_batch in train_loader:
                u_batch, i_batch, r_batch = u_batch.to(device), i_batch.to(device), r_batch.to(device)

                optimizer.zero_grad()
                preds = self.forward(u_batch, i_batch)
                loss = criterion(preds, r_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * u_batch.size(0)
                
            avg_train_loss = train_loss / len(train_loader.dataset)
            self.training_losses.append(avg_train_loss)

            # Validation loop
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for u_batch, i_batch, r_batch in val_loader:
                    u_batch, i_batch, r_batch = u_batch.to(device), i_batch.to(device), r_batch.to(device)
                    preds = self.forward(u_batch, i_batch)
                    val_loss += criterion(preds, r_batch).item() * u_batch.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            self.validation_losses.append(avg_val_loss)

            # Print training and validation loss
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # torch.save(self.model.state_dict(), "best_model.pth")
                self.save_model_and_params(path="../data/models/best_model.pth")
                print(f"New best model saved with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")

    def save_model_and_params(self, path="../data/models/best_model.pth"):
        """
        Save model's state_dict and other essential attributes, including default_user_profile.
        """
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "item_features": self.item_features,
            "user_profiles": self.user_profiles,
            "default_user_profile": self.default_user_profile,
            "user2idx": self.user2idx,
            "item2idx": self.item2idx
        }
        torch.save(save_dict, path)
        print(f"Model and parameters saved to {path}.")

    def load_model_and_params(self, path="../data/models/best_model.pth"):
        """
        Load model's state_dict and other essential attributes, including default_user_profile.
        """
        checkpoint = torch.load(path)
        
        # Load attributes
        self.item_features = checkpoint["item_features"]
        self.user_profiles = checkpoint["user_profiles"]
        self.default_user_profile = checkpoint["default_user_profile"]  # Added
        self.user2idx = checkpoint["user2idx"]
        self.item2idx = checkpoint["item2idx"]

        # Rebuild the model architecture 
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Model and parameters loaded from {path}.")
