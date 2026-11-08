# IMDb Sentiment Analysis using DistilBERT + TensorFlow

A notebook-based Natural Language Processing (NLP) project that fine-tunes a pretrained **DistilBERT Transformer** for binary sentiment classification of IMDb movie reviews. The project explores transfer learning, compares Transformer-based classification against a traditional TF-IDF + Logistic Regression baseline, and performs threshold optimization and error analysis.

## Tech Stack

- **Deep Learning:** TensorFlow, Keras
- **NLP:** DistilBERT, KerasHub
- **Machine Learning:** Scikit-learn, Logistic Regression, TF-IDF
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib

This is a TensorFlow/Keras implementation. It does not use PyTorch or Hugging Face Trainer.

## Dataset

The project uses the IMDb movie review dataset available through `keras.datasets.imdb`.

Reviews are reconstructed from their word-index representations and processed using the DistilBERT tokenizer and preprocessing pipeline.

A stratified subset is used to keep model fine-tuning computationally manageable:

| Dataset | Samples |
|---|---:|
| Training | 6,400 |
| Validation | 1,600 |
| Testing | 5,000 |
| **Total** | **13,000** |

The training and validation samples are selected from the official training split, while the test samples come from the official test split.

## Model Performance

The fine-tuned DistilBERT model was evaluated on 5,000 held-out IMDb reviews after two training epochs.

### DistilBERT results

| Metric | Default Threshold (0.50) | Tuned Threshold (0.75) |
|---|---:|---:|
| Accuracy | 88.28% | **89.46%** |
| ROC-AUC | 0.9619 | 0.9619 |
| Precision | 84.15% | 91.02% |
| Recall | 94.32% | 87.56% |
| F1-score | 88.95% | 89.26% |
| Negative Recall | 82.24% | 91.36% |

The decision threshold was optimized using validation-set macro-F1 and then applied to the held-out test set. This improved overall accuracy and balanced performance between positive and negative reviews.

### Baseline comparison

A TF-IDF + Logistic Regression model was implemented as a traditional machine learning baseline.

| Model | Test Accuracy | ROC-AUC |
|---|---:|---:|
| TF-IDF + Logistic Regression | 88.12% | 0.9525 |
| DistilBERT (default threshold) | 88.28% | 0.9619 |
| DistilBERT (tuned threshold) | **89.46%** | **0.9619** |

## Project Workflow

1. Load and reconstruct IMDb movie reviews from word indices.
2. Create stratified training, validation and test samples.
3. Build a TF-IDF + Logistic Regression baseline.
4. Load pretrained DistilBERT using KerasHub.
5. Fine-tune the Transformer using TensorFlow/Keras.
6. Monitor training and validation accuracy and loss.
7. Evaluate using accuracy, precision, recall, F1-score and ROC-AUC.
8. Analyze predictions using a confusion matrix.
9. Optimize the classification threshold using validation macro-F1.
10. Compare baseline and Transformer performance.
11. Investigate misclassified reviews, high-confidence errors and error patterns.
12. Generate sentiment predictions for custom movie reviews.
13. Save the trained model, evaluation metrics and analysis outputs.

## Project Structure

```text
IMDb_DistilBERT_TensorFlow/
│
├── sentiment_analysis_distilbert_tensorflow.ipynb
├── requirements.txt
├── README.md
│
└── outputs/
    ├── distilbert_imdb.keras
    ├── metrics.json
    ├── model_comparison.csv
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── misclassified_reviews.csv
    ├── high_confidence_errors.csv
    └── error_analysis_summary.csv
```

The outputs directory and its files are generated when the notebook is executed.

## How to Run

A GPU-enabled environment such as Google Colab with a T4 GPU is recommended for fine-tuning DistilBERT.

1. Clone the repository.
2. Open the notebook in Google Colab or Jupyter Notebook.
3. Select a GPU runtime.
4. Install the dependencies using the installation cell or:

```bash
pip install -r requirements.txt
```

5. Execute the notebook cells sequentially.

The IMDb dataset downloads automatically during the first execution.

## Limitations

- The experiment uses a subset of the IMDb dataset rather than the complete dataset.
- Reviews are reconstructed from word indices, resulting in lowercase text without original punctuation.
- DistilBERT processes a maximum sequence length of 256 tokens, so longer reviews are truncated.
- The TF-IDF baseline uses full reconstructed reviews, whereas DistilBERT uses truncated sequences.
- Results are based on a single training run and may vary with random initialization and training conditions.
- Threshold optimization is performed on a relatively small validation set.
- Error-category analysis uses simple keyword-based rules and cannot reliably identify complex linguistic phenomena such as sarcasm.
- No production deployment or API integration is included.

## Key Takeaway

This project demonstrates the application of transfer learning using pretrained Transformer architectures for NLP classification. It also highlights the importance of baseline comparisons, class-wise evaluation, decision-threshold optimization and error analysis when developing machine learning models.
