
# Early Stopping in NLP Sentiment Analysis

**Project Title:** Early Stopping in NLP Sentiment Models  
**Objective:** Implement early stopping during training of a text classifier to prevent overfitting on sparse representations (TF-IDF).

## Overview

This project trains a high-performance sentiment classifier on the **Sentiment140** dataset (Twitter sentiment) using:
- Sparse TF-IDF representations (word + character n-grams)
- Logistic Regression / SGDClassifier
- Proper early stopping with patience and best-weight restoration
- Live interactive loss curves showing the stopping point

**Key Results (on real data):**
- Accuracy: **~81%**
- F1-Score: **~0.81**
- Clear early stopping behavior with live loss visualization

> Note: 81% is a strong and realistic result for classical (non-transformer) models on the noisy Sentiment140 dataset. State-of-the-art classical approaches typically range between 80–83%.

## Features

- Real Sentiment140 dataset (1.6 million tweets)
- Balanced sampling
- Advanced text preprocessing (negation handling, elongated words, hashtag cleaning)
- TF-IDF with n-grams (sparse representations)
- Early stopping with patience and best model restoration
- Live updating loss curve during training
- Model serialization (`.joblib`)
- One-file, ready-to-run script

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

## How to Run

1. Make sure you have Python 3.9+ installed.
2. Install the dependencies (see above).
3. Run the main script:

```bash
python train.py
```

The script will:
- Automatically download the Sentiment140 dataset (first run only)
- Train the model
- Show live early-stopping loss curves
- Print final Accuracy and F1-Score
- Save the model and plot

At the end, press **Enter** to exit.

## Project Structure

```
.
├── train.py                     # Main training script (single file)
├── training.1600000.processed.noemoticon.csv   # Dataset (downloaded automatically)
├── *.joblib                     # Saved models
├── early_stopping*.png          # Loss curve plots
├── README.md
└── .gitignore
```

## Key Concepts Demonstrated

- Early Stopping (patience + restore best weights)
- Sparse Representations (TF-IDF)
- Parameter tuning (C, n-grams, regularization)
- Live training visualization
- Handling of real-world noisy Twitter data

## Expected Output Example

```
FINAL ACCURACY : 81.XX%
FINAL F1-SCORE : 0.81XX

              precision    recall  f1-score   support
    Negative       0.82      0.80      0.81     ...
    Positive       0.81      0.82      0.81     ...
```

## Notes

- First run downloads ~230 MB of data.
- Subsequent runs are much faster (uses local CSV).
- The live plot window must stay open during training.

## Author

Bhumika P – Deep Learning Project
```