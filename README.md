# Sentiment Analysis Project

This project implements sentiment analysis using multiple approaches, including deep learning with PyTorch and traditional machine learning methods.

## Project Structure

```
Sentiment-Analysis/
│
├── datasets/
│   └── IMDB-Dataset.csv
│
├── __pycache__/
├── .venv/
│
├── config.py
├── data_processing.py
├── experiment.ipynb
├── models.py
├── requirements.txt
├── training.py
├── utils.py
└── visualize.py
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/Valiev-Koyiljon/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The main experiment is conducted in the `experiment.ipynb` Jupyter notebook. To run the experiments:

1. Ensure you have Jupyter installed:
   ```
   pip install jupyter
   ```

2. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

3. Open `experiment.ipynb` in the Jupyter interface and run the cells to execute the sentiment analysis experiments.

## Project Components

- `config.py`: Contains configuration parameters for the models and data processing.
- `data_processing.py`: Handles data loading, preprocessing, and preparation.
- `models.py`: Defines the model architectures (likely including RNN, SGD, and Multinomial Naive Bayes).
- `training.py`: Contains training loops and procedures.
- `utils.py`: Utility functions used across the project.
- `visualize.py`: Functions for visualizing results and model performance.

## Data

The project uses the IMDB Dataset, located at `datasets/IMDB-Dataset.csv`. This dataset contains movie reviews labeled with sentiment (positive/negative).

## Models

The project implements multiple models for sentiment analysis:
1. Recurrent Neural Network (RNN) using PyTorch
2. Stochastic Gradient Descent (SGD) classifier
3. Multinomial Naive Bayes (MNB) classifier

Refer to `models.py` for the specific implementations of these models.

## Evaluation

The models are evaluated in the `experiment.ipynb` notebook. This includes metrics such as accuracy, precision, recall, and F1-score, as well as visualizations of the results.

## Visualization

The `visualize.py` file contains functions for creating visualizations of the model performance and results. These visualizations can be generated and viewed within the `experiment.ipynb` notebook.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

---

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/Valiev-Koyiljon/Sentiment-Analysis).
