import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from config import config

def set_seed(seed=config.RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def train_val_test_split(data, labels):
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=(config.VAL_SIZE + config.TEST_SIZE), stratify=labels, random_state=config.RANDOM_SEED
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        test_data, test_labels, test_size=config.TEST_SIZE/(config.VAL_SIZE + config.TEST_SIZE), 
        stratify=test_labels, random_state=config.RANDOM_SEED
    )
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

def create_rnn_dataloaders(X, y, batch_size, shuffle=False):
    """Create DataLoader objects for RNN models"""
    if sp.issparse(X):
        X = X.tocsr()  # Convert to CSR format for efficient row slicing
        dataset = SparseDataset(X, y)
    else:
        dataset = TensorDataset(torch.LongTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class SparseDataset(torch.utils.data.Dataset):
    """Custom Dataset for sparse matrices"""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_row = torch.LongTensor(self.X[idx].toarray().squeeze())
        y_item = torch.FloatTensor([self.y[idx]])
        return X_row, y_item