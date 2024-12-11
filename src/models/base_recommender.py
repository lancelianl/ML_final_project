import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict

class BaseRecommender(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, user_ids, item_ids):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement the forward method.")

    def fit(self, train_df):
        # Optional method for preprocessing, training, or fitting parameters.
        pass