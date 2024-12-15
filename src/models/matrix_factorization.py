import torch
import torch.nn as nn
from models.base_recommender import BaseRecommender

class MatrixFactorizationModel(BaseRecommender):
    def __init__(self, num_users, num_items, embedding_dim=32):
        """
        Matrix Factorization Model for Collaborative Filtering
        Parameters:
        - num_users: Total number of unique users
        - num_items: Total number of unique items (anime)
        - embedding_dim: Size of latent factors
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Bias terms for users and items
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        """
        Predict ratings for user-item pairs.
        Parameters:
        - user_ids: Tensor of user indices
        - item_ids: Tensor of item indices
        Returns:
        - Predicted ratings (Tensor)
        """
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)

        # Dot product of user and item embeddings
        dot_product = (user_vecs * item_vecs).sum(dim=1, keepdim=True)

        # Add bias terms
        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)

        # Final prediction
        prediction = dot_product + user_bias + item_bias
        return prediction.squeeze()

    def fit(self, train_loader, val_loader=None, num_epochs=10, lr=0.01, device='cpu', early_stopping_patience=3):
        """
        Train the Matrix Factorization model with early stopping.
        Parameters:
        - early_stopping_patience: Number of epochs to wait for validation loss improvement before stopping.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for batch in train_loader:
                user_ids, item_ids, ratings = batch
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)

                optimizer.zero_grad()
                predictions = self(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

            # Validation step
            if val_loader:
                val_loss = self.evaluate(val_loader, criterion, device)
                print(f"Validation MSE: {val_loss:.4f}")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Optionally save the best model weights
                    best_model_state = self.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping triggered.")
                        # Load the best model weights (optional)
                        self.load_state_dict(best_model_state)
                        break


    def evaluate(self, data_loader, criterion, device='cpu'):
        """
        Evaluate the model on a given dataset.
        Parameters:
        - data_loader: DataLoader for evaluation data
        - criterion: Loss function (e.g., MSELoss)
        - device: 'cpu' or 'cuda'
        Returns:
        - MSE score
        """
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                user_ids, item_ids, ratings = batch
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
                predictions = self(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                total_loss += loss.item()

        return total_loss / len(data_loader)
