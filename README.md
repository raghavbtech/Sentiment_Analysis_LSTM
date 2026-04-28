# Reel Feelings — Movie Sentiment Analyzer

A sentiment analysis app for movie reviews, built with an LSTM model trained on the IMDB dataset. Runs as a Streamlit web app.

## What it does

Takes a movie review as input and predicts whether the sentiment is positive or negative, along with a confidence score.

## Project structure

```
├── app.py              # Streamlit UI
├── model.py            # LSTM model definition
├── LSTM.ipynb          # Training notebook
├── LSTMmodel.pth       # Trained model weights
├── tokenizer.pkl       # Fitted Keras tokenizer
└── IMDB Dataset.csv    # Training data (50k reviews)
```

## Model

- **Architecture:** Embedding → LSTM → Linear
- **Embedding size:** 128
- **Hidden size:** 128
- **Vocab size:** 5001
- **Output:** Sigmoid probability (>0.5 = positive)

## Setup

```bash
pip install streamlit torch tensorflow
```

```bash
streamlit run app.py
```

## Dataset

[IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — 50,000 labeled reviews, balanced between positive and negative.
