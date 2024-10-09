import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from models import RNN, get_multinomial_nb, get_sgd_classifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from training import train_model, train_sklearn_model, evaluate_sklearn_model, create_dataloaders, evaluate
from data_processing import load_and_preprocess_data, build_vocabulary, vectorize_data, process_in_chunks
from visualize import plot_confusion_matrix
from tqdm import tqdm
from config import config

def train_model(model, train_iterator, valid_iterator, optimizer, criterion, n_epochs=config.NUM_EPOCHS):
    model = model.to(config.DEVICE)
    best_valid_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        
        for i, (text, labels) in enumerate(tqdm(train_iterator, desc=f"Epoch {epoch+1}/{n_epochs}")):
            text, labels = text.to(config.DEVICE), labels.to(config.DEVICE).squeeze(1)
            
            optimizer.zero_grad()
            predictions = model(text)
            loss = criterion(predictions, labels)
            loss = loss / config.ACCUMULATION_STEPS  # Normalize the loss
            loss.backward()
            
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config.ACCUMULATION_STEPS
            train_acc += ((predictions > 0) == labels).float().mean().item()
        
        # Perform the last optimization step if needed
        if (i + 1) % config.ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss /= len(train_iterator)
        train_acc /= len(train_iterator)
        
        valid_loss, valid_acc, _, _ = evaluate(model, valid_iterator, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")



def train_rnn(train_iterator, val_iterator, test_iterator, vocab_size):
    print("Initializing and training RNN model...")
    rnn_model = RNN(
        vocab_size, 
        config.EMBEDDING_DIM, 
        config.HIDDEN_DIM, 
        config.OUTPUT_DIM, 
        config.N_LAYERS, 
        config.BIDIRECTIONAL, 
        config.DROPOUT
    )
    optimizer = optim.Adam(rnn_model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    train_model(rnn_model, train_iterator, val_iterator, optimizer, criterion)

    print("Evaluating RNN model...")
    rnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    test_loss, test_acc, y_pred, y_true = evaluate(rnn_model, test_iterator, criterion)
    
    # Calculate additional metrics
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"RNN Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    plot_confusion_matrix(cm, classes=['Negative', 'Positive'])

def train_mnb(X_train, train_labels, X_test, test_labels):
    print("Training and evaluating Multinomial Naive Bayes...")
    mnb = get_multinomial_nb()
    mnb = train_sklearn_model(mnb, X_train, train_labels)
    mnb_accuracy, mnb_report, mnb_cm = evaluate_sklearn_model(mnb, X_test, test_labels)
    print(f"Multinomial Naive Bayes Accuracy: {mnb_accuracy:.4f}")
    print("Classification Report:")
    print(mnb_report)
    print("Confusion Matrix:")
    print(mnb_cm)
    plot_confusion_matrix(mnb_cm, classes=['Negative', 'Positive'])

def train_sgd(X_train, train_labels, X_test, test_labels):
    print("Training and evaluating SGD Classifier...")
    sgd = get_sgd_classifier()
    sgd = train_sklearn_model(sgd, X_train, train_labels)
    sgd_accuracy, sgd_report, sgd_cm = evaluate_sklearn_model(sgd, X_test, test_labels)
    print(f"SGD Classifier Accuracy: {sgd_accuracy:.4f}")
    print("Classification Report:")
    print(sgd_report)
    print("Confusion Matrix:")
    print(sgd_cm)
    plot_confusion_matrix(sgd_cm, classes=['Negative', 'Positive'])


def evaluate(model, iterator, criterion):
    model = model.to(config.DEVICE)
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for text, labels in iterator:
            text, labels = text.to(config.DEVICE), labels.to(config.DEVICE).squeeze(1)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            
            predictions = (predictions > 0).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(iterator)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels



def train_sklearn_model(model, X_train, y_train):
    return model.fit(X_train, y_train)

def evaluate_sklearn_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm

def create_dataloaders(train_data, train_labels, val_data, val_labels, test_data, test_labels):
    train_dataset = TensorDataset(torch.LongTensor(train_data), torch.FloatTensor(train_labels))
    val_dataset = TensorDataset(torch.LongTensor(val_data), torch.FloatTensor(val_labels))
    test_dataset = TensorDataset(torch.LongTensor(test_data), torch.FloatTensor(test_labels))
    
    train_iterator = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_iterator = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_iterator = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    return train_iterator, val_iterator, test_iterator