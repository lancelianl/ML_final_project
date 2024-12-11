import torch
import torch.nn as nn

def evaluate_mse(model, dataloader, device='cpu'):
    """
    Evaluate the model using MSE on the given dataloader.
    
    Parameters:
    - model: an instance of BaseRecommender (or subclass) which implements forward(user_ids, item_ids).
    - dataloader: a DataLoader for either validation or test set.
    - device: 'cpu' or 'cuda', depending on where you want to run the evaluation.
    
    Returns:
    - mse: float, the mean squared error over all samples in the dataloader.
    """
    model.eval()  # Set model to evaluation mode
    mse_loss = nn.MSELoss(reduction='sum')  # sum up for all samples, then divide by count
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            # batch could be (user_ids, anime_ids, ratings, ...) depending on dataset configuration
            user_ids = batch[0].to(device)
            anime_ids = batch[1].to(device)
            ratings = batch[2].to(device)

            # Forward pass
            preds = model(user_ids, anime_ids)

            # Compute squared error manually or use MSELoss
            # Since we want total sum, we use MSELoss with reduction='sum'
            loss = mse_loss(preds, ratings)

            total_loss += loss.item()
            total_count += len(ratings)

    mse = total_loss / total_count if total_count > 0 else 0.0
    return mse

# if __name__ == "__main__":
#     # Example usage:
#     # Assuming you have a trained model and a test_loader DataLoader
#     # from your main script, you might do something like:
#     #
#     # from your_model_script import GenreBaselineModel
#     # from your_preprocessing_script import preprocess_data
#     # from anime_data_script import AnimeData
#     #
#     # anime_data = AnimeData('data/raw/anime.csv')
#     # train_dataset, test_dataset = preprocess_data(filepath='data/raw/animelist.csv')
#     # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#     #
#     # model = GenreBaselineModel(anime_data)
#     # model.load_state_dict(torch.load('path_to_saved_weights.pt'))
#     #
#     # mse_score = evaluate_mse(model, test_loader, device='cpu')
#     # print("Test MSE:", mse_score)
#     pass
