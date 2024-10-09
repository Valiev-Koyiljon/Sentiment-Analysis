
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from config import config

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=config.WORDCLOUD_WIDTH, 
                          height=config.WORDCLOUD_HEIGHT, 
                          max_font_size=config.WORDCLOUD_MAX_FONT_SIZE).generate(text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()

def plot_confusion_matrix(cm, classes=['Negative', 'Positive']):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()