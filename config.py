import torch

class Config:
    # Data parameters
    DATA_FILE = 'datasets/IMDB-Dataset.csv'
    MAX_VOCAB_SIZE = 10000  # Reduced from 25000
    MAX_SEQUENCE_LENGTH = 100  # Reduced from 200
    
    # Data split parameters
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    TEST_SIZE = 0.1
    
    # Model parameters
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128  # Reduced from 256
    OUTPUT_DIM = 1
    N_LAYERS = 2  # Reduced from 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    # Training parameters
    BATCH_SIZE = 32  # Reduced from 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    ACCUMULATION_STEPS = 2  # New parameter for gradient accumulation
    
    # Other parameters
    RANDOM_SEED = 42
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Paths
    MODEL_SAVE_PATH = 'best_model/best_model.pt'
    
    # Preprocessing parameters
    REMOVE_DIGITS = True
    STEM_WORDS = True
    REMOVE_STOPWORDS = True
    
    # Vectorization parameters
    USE_TFIDF = True
    NGRAM_RANGE = (1, 2)
    
    # SGD Classifier parameters
    SGD_LOSS = 'hinge'
    SGD_PENALTY = 'l2'
    SGD_ALPHA = 0.0001
    SGD_MAX_ITER = 1000
    
    # Multinomial Naive Bayes parameters
    MNB_ALPHA = 1.0
    
    # Visualization parameters
    WORDCLOUD_WIDTH = 800
    WORDCLOUD_HEIGHT = 400
    WORDCLOUD_MAX_FONT_SIZE = 110

config = Config()