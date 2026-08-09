# Movie-Reviews-Sentiment-Analysis-Using-LLM
Built a PyTorch Transformer Encoder for IMDB sentiment classification, including text preprocessing, tokenization, positional embeddings, multi-head attention, and train/validation/test pipelines. Achieved ~83% test accuracy, outperforming the LSTM baseline.

# Sentiment Analysis with Transformer Encoder

This project implements a **Transformer Encoder-based sentiment classification model in PyTorch** to predict whether an IMDB movie review expresses positive or negative sentiment.

## Project Overview

The model applies a Transformer architecture to capture both word meaning and contextual relationships within movie reviews. Unlike sequential models such as LSTMs, the Transformer uses **multi-head self-attention** to evaluate relationships between words across the entire sequence.

The complete pipeline includes:

* Cleaning and preprocessing IMDB movie reviews
* Converting text into integer-based tokens
* Encoding positive/negative labels
* Removing invalid reviews and handling sequence-length outliers
* Padding and truncating reviews to a fixed sequence length of **400 tokens**
* Splitting the dataset into training, validation, and test sets
* Creating PyTorch `DataLoader` pipelines
* Building token and positional embeddings
* Implementing a custom Transformer Encoder
* Training and evaluating the sentiment classifier
* Performing inference on new movie reviews

## Dataset

The project uses **25,000 labeled IMDB movie reviews** after removing one zero-length observation.

The dataset is divided into:

| Dataset    | Samples | Percentage |
| ---------- | ------: | ---------: |
| Training   |  20,000 |        80% |
| Validation |   2,500 |        10% |
| Test       |   2,500 |        10% |

Each review is converted into a fixed-length sequence of **400 tokens** through padding or truncation.

## Model Architecture

The sentiment classifier consists of:

**1. Token Embeddings**
Words are converted from integer tokens into 512-dimensional embedding vectors.

**2. Positional Embeddings**
Position information is added to token embeddings so the model can distinguish the order of words within a review.

**3. Transformer Encoder**

The encoder contains:

* Multi-head self-attention
* 2 attention heads
* Residual connections
* Layer normalization
* Feed-forward neural network
* ReLU activation

The feed-forward network expands the embedding dimension from **512 → 1024 → 512**.

**4. Global Max Pooling**

The Transformer output across all 400 tokens is reduced into a single feature representation using max pooling.

**5. Classification Layer**

The pooled representation passes through:

`Dropout (0.2) → Linear Layer → Sigmoid`

The final output represents the probability that a review has positive sentiment.

## Training

The model was trained using:

* **Optimizer:** Adam
* **Loss Function:** Binary Cross Entropy (`BCELoss`)
* **Batch Size:** 32
* **Epochs:** 2
* **Sequence Length:** 400
* **Embedding Dimension:** 512
* **Attention Heads:** 2
* **Feed-Forward Dimension:** 1024

The model began showing signs of overfitting after approximately two epochs.

## Results

| Metric                        |     Result |
| ----------------------------- | ---------: |
| Validation Accuracy – Epoch 1 |     84.33% |
| Validation Accuracy – Epoch 2 |     83.05% |
| Test Accuracy                 | **83.05%** |
| Test Loss                     |  **0.378** |

The Transformer achieved approximately **83% test accuracy**, outperforming the LSTM baseline referenced in the exercise.

## Example Prediction

```text
Input:
"The worst movie I have seen; acting was terrible and I want my money back."

Prediction Probability: 0.003335
Prediction: Negative Review
```

```text
Input:
"The cinematography is excellent."

Prediction Probability: 0.956132
Prediction: Positive Review
```

## Technologies

* Python
* PyTorch
* NumPy
* Transformer Encoder
* Multi-Head Attention
* Natural Language Processing
* Deep Learning
* Sentiment Analysis

## Key Takeaways

This project demonstrates how a Transformer Encoder can be built from core PyTorch components and applied to binary text classification. It covers the complete NLP workflow from raw text preprocessing and positional encoding to self-attention, model training, evaluation, and inference on unseen reviews.
