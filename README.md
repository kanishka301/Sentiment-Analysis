# IMDb Sentiment Analysis using DistilBERT + TensorFlow

A notebook-first NLP project that fine-tunes pretrained DistilBERT for positive/negative movie-review sentiment classification.

## Stack

- TensorFlow / Keras
- KerasHub (DistilBERT preset and text preprocessing)
- scikit-learn, pandas, matplotlib

This project does **not** use PyTorch or Hugging Face Trainer.

## Dataset

IMDb reviews are loaded with `keras.datasets.imdb` and converted back to text for the DistilBERT tokenizer. The data downloads automatically on the first run.

The notebook uses a stratified subset:
- 8,000 training reviews (80% train / 20% validation)
- 5,000 reviews from the official test split

Sizes are configurable in the notebook and saved in `outputs/metrics.json`.

## Results

Test set of 5,000 reviews, 2 epochs, sequence length 128:

| Accuracy | ROC-AUC | Negative recall | Positive recall |
|---|---|---|---|
| 85.7% | 0.943 | 0.784 | 0.930 |

The model is stronger on positive reviews than negative ones.

## Project structure

```text
IMDb_DistilBERT_TensorFlow/
├── sentiment_analysis_distilbert_tensorflow.ipynb
├── requirements.txt
├── README.md
└── outputs/        # created when the notebook runs
```

## Run

A GPU runtime such as Google Colab T4 is strongly recommended.

1. Open the notebook in Google Colab or Jupyter and select a GPU runtime.
2. Run the install cell, then the remaining cells from top to bottom.

Local setup: `pip install -r requirements.txt`

## Workflow

1. Load IMDb reviews and rebuild text from word indices
2. Create train / validation / test samples
3. Load pretrained DistilBERT with KerasHub
4. Fine-tune with TensorFlow/Keras
5. Plot training and validation curves
6. Evaluate with accuracy, precision, recall, F1 and ROC-AUC
7. Plot a confusion matrix
8. Inspect misclassified and high-confidence incorrect reviews
9. Predict sentiment for custom reviews
10. Save model and metrics

## Outputs

`distilbert_imdb.keras` (git-ignored), `metrics.json`, `training_curves.png`, `confusion_matrix.png`, `misclassified_reviews.csv`, `high_confidence_errors.csv`

## Limitations

A subset of IMDb is used, reviews are cut to 128 tokens and rebuilt from word indices (lowercase, no punctuation), and results come from a single run.
