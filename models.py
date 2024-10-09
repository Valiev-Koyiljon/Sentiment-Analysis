import torch
import torch.nn as nn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from config import config

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        
        # Pack the embedded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, (text != 0).sum(1).cpu(), batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden).squeeze(1)  # Remove the last dimension


def get_multinomial_nb():
    return MultinomialNB(alpha=config.MNB_ALPHA)

def get_sgd_classifier():
    return SGDClassifier(loss=config.SGD_LOSS, 
                         penalty=config.SGD_PENALTY, 
                         alpha=config.SGD_ALPHA, 
                         max_iter=config.SGD_MAX_ITER, 
                         random_state=config.RANDOM_SEED)