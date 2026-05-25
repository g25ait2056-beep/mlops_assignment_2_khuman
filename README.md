---
library_name: transformers
tags: []
---
library_name: transformers
language:
- en
license: apache-2.0
base_model: distilbert-base-cased
tags:
- text-classification
- distilbert
- book-reviews
- genre-classification
- mlops
datasets:
- UCSD-Goodreads
metrics:
- accuracy
- f1
pipeline_tag: text-classification
---
# Model Card for distilbert-goodreads-genres
This model is a fine-tuned version of `distilbert-base-cased` that classifies short-to-medium length book reviews into one of eight literary genres. It was developed as part of an MLOps assignment to demonstrate experiment tracking (Weights & Biases) and model deployment (Hugging Face Hub).
## Model Details
### Model Description
A DistilBERT-based sequence classification model fine-tuned on a sampled subset of the UCSD Goodreads Reviews dataset. Given a free-text book review, the model predicts the genre of the reviewed book across eight classes: Children, Comics & Graphic, Fantasy & Paranormal, History & Biography, Mystery/Thriller/Crime, Poetry, Romance, and Young Adult.

- **Developed by:** Laishram Khumanleima Chanu (g25ait2056)
- **Funded by:** IIT Jodhpur
- **Shared by:** Laishram Khumanleima Chanu
- **Model type:** Transformer-based text classification (DistilBERT for sequence classification)
- **Language(s) (NLP):** English
- **License:** Apache 2.0 (inherited from the base model)
- **Finetuned from model:** [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)

### Model Sources

- **Repository:** [LKHUMANLEIMA/distilbert-goodreads-genres](https://huggingface.co/LKHUMANLEIMA/distilbert-goodreads-genres)

## Uses

### Direct Use

The model is intended for classifying short-to-medium length book reviews into one of eight genres: Children, Comics & Graphic, Fantasy & Paranormal, History & Biography, Mystery/Thriller/Crime, Poetry, Romance, and Young Adult. It is suitable for educational, exploratory, and demonstrative purposes in the context of literary text classification.

### Downstream Use

The model can be used as a starting point for further fine-tuning on related book-review classification tasks, or as a feature extractor for downstream analyses such as recommendation systems and review-corpus exploration.

### Out-of-Scope Use

The model should not be used for high-stakes decision-making or for analyzing text outside the literary review domain. It is not suitable for:
- Classifying texts in languages other than English.
- Classifying non-review text (news articles, social media posts, fiction itself, etc.).
- Predicting genres not in the training label set.
- Any application where a misclassification could cause harm to users or third parties.

## Bias, Risks, and Limitations

- The training data is a small random sample (1,000 reviews per genre) from the UCSD Goodreads corpus and may not reflect the full diversity of book reviews.
- Goodreads reviews are user-generated and reflect the demographic biases of the Goodreads user base.
- The model may confuse genres with overlapping themes (e.g., Fantasy & Paranormal vs. Young Adult, Children vs. Young Adult).
- The model only handles English text and was trained on reviews truncated to 512 tokens.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases, and limitations of the model. 
## How to Get Started with the Model
```python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

model_name = "LKHUMANLEIMA/distilbert-goodreads-genres"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

text = "A beautifully written romance set in 19th century England..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_id = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_id]
print(predicted_label)
```
## Training Details
### Training Data
The training data is derived from the [UCSD Goodreads Reviews dataset](https://mengtingwan.github.io/data/goodreads.html), specifically the per-genre review files. For each of the eight target genres, the first 10,000 reviews were streamed from the source, and a random sample of 2,000 reviews per genre was retained. From this sample, 1,000 reviews per genre were used in the modeling pipeline: 800 for training and 200 for testing, giving:

- **Training set:** 6,400 reviews (800 × 8 genres)
- **Test set:** 1,600 reviews (200 × 8 genres)

Labels: `children`, `comics_graphic`, `fantasy_paranormal`, `history_biography`, `mystery_thriller_crime`, `poetry`, `romance`, `young_adult`.

### Training Procedure

The model was fine-tuned using the Hugging Face `Trainer` API, with experiment tracking via Weights & Biases (project: `distilbert-goodreads-genres`, run: `distilbert-run-1`).

#### Preprocessing

- Reviews tokenized with `DistilBertTokenizerFast` (cased).
- Truncation and padding applied to a maximum length of 512 tokens.
- String genre labels mapped to integer ids using a `label2id` / `id2label` dictionary built from unique training labels.

#### Training Hyperparameters

- **Base model:** `distilbert-base-cased`
- **Number of epochs:** 3
- **Per-device train batch size:** 16
- **Per-device eval batch size:** 32
- **Learning rate:** 5e-5 (Adam)
- **Warmup steps:** 100
- **Weight decay:** 0.01
- **Max sequence length:** 512
- **Evaluation strategy:** per epoch
- **Save strategy:** per epoch
- **Load best model at end:** True
- **Training regime:** fp32

#### Speeds, Sizes, Times

Training was run for 3 epochs over 6,400 training examples (global_step = 600 with batch size 16). Reported metrics from the final `Trainer` output:

| Metric                    | Value                  |
|---------------------------|------------------------|
| Total training runtime    | 563.36 s (~9.4 min)    |
| Train samples / second    | 34.08                  |
| Train steps / second      | 1.065                  |
| Total FLOPs               | 2.544 × 10¹⁵           |
| Global step               | 600                    |
| Final training loss       | 2.2209                 |
| Epochs                    | 3.0                    |

Evaluation on the test set:

| Metric                   | Value      |
|--------------------------|------------|
| Eval runtime             | 14.89 s    |
| Eval samples / second    | 107.44     |
| Eval steps / second      | 1.679      |
| Checkpoint size          | ~263 MB (DistilBERT base weights + 8-class classification head) |

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

A held-out test set of 1,600 reviews (200 per genre) sampled from the same UCSD Goodreads source as the training data, disjoint from the training partition.

#### Factors

Performance is disaggregated by **book genre**, the only metadata attached to each review in this pipeline. The test set is perfectly class-balanced (200 reviews per genre), so per-class scores are directly comparable. Per-class precision, recall, and F1 on the test set:

| Genre                   | Precision | Recall | F1   | Support |
|-------------------------|-----------|--------|------|---------|
| children                | 0.71      | 0.60   | 0.65 | 200     |
| comics_graphic          | 0.78      | 0.76   | 0.77 | 200     |
| fantasy_paranormal      | 0.50      | 0.40   | 0.44 | 200     |
| history_biography       | 0.56      | 0.56   | 0.56 | 200     |
| mystery_thriller_crime  | 0.54      | 0.62   | 0.58 | 200     |
| poetry                  | 0.70      | 0.78   | 0.74 | 200     |
| romance                 | 0.52      | 0.61   | 0.56 | 200     |
| young_adult             | 0.34      | 0.34   | 0.34 | 200     |
| **macro avg**           | 0.58      | 0.58   | 0.58 | 1600    |
| **weighted avg**        | 0.58      | 0.58   | 0.58 | 1600    |

The model performs best on visually and stylistically distinctive genres (**comics_graphic**, F1 0.77; **poetry**, F1 0.74), and worst on **young_adult** (F1 0.34) and **fantasy_paranormal** (F1 0.44), which are thematically broad and overlap heavily with other genres (notably mystery_thriller_crime, romance, and children). For reference, the TF-IDF + logistic regression baseline reported in the notebook reached only 0.53 weighted F1, so fine-tuning DistilBERT improves overall F1 by about 5 percentage points, with the largest absolute gains on **poetry** (+0.09 F1) and **comics_graphic** (+0.08 F1).

Other potentially relevant factors (review length, reviewer identity, publication year, rating) are present in the source UCSD Goodreads data but were not used or evaluated in this pipeline.

#### Metrics

- **Accuracy** — proportion of reviews assigned the correct genre.
- **F1 (weighted)** — class-frequency-weighted F1, reported because the test set is balanced but the metric is robust if class balance changes.
- **Eval loss** — cross-entropy loss on the test set.

### Results

Training progression across epochs:

| Epoch | Training Loss | Validation Loss | Accuracy | F1     |
|-------|---------------|-----------------|----------|--------|
| 1     | 2.5425        | 2.5051          | 0.5544   | 0.5448 |
| 2     | 2.0217        | 2.3416          | 0.5819   | 0.5799 |
| 3     | 1.3738        | 2.4122          | 0.5856   | 0.5843 |

Best-checkpoint evaluation metrics (loaded via `load_best_model_at_end=True`):

| Metric            | Value   |
|-------------------|---------|
| eval/accuracy     | 0.58188 |
| eval/f1           | 0.57991 |
| eval/loss         | 2.34165 |
| eval/runtime (s)  | 14.8917 |
| samples / second  | 107.442 |
| steps / second    | 1.679   |

#### Summary

The fine-tuned DistilBERT model achieves roughly 58% accuracy and 0.58 weighted F1 on an 8-way genre classification task, comfortably above the 12.5% random baseline and ~5 points above the TF-IDF + logistic regression baseline. Performance is limited by the small training set (6,400 reviews) and overlap between thematically similar genres; the rising training-loss–to-validation-loss gap by epoch 3 suggests mild overfitting and that additional regularization or more data would likely improve generalization.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GPU (CUDA), via Kaggle Notebooks
- **Hours used:** Approximately 0.16 GPU-hours (training: 563 s; evaluation: ~15 s)
- **Cloud Provider:** Kaggle (Google Cloud underlying infrastructure)
- **Compute Region:** Not specified
- **Carbon Emitted:** Not measured

## Technical Specifications

### Model Architecture and Objective

DistilBERT (a distilled, 6-layer transformer derived from BERT-base) with a sequence classification head on top of the pooled `[CLS]` representation. The objective is multi-class cross-entropy over 8 genre labels.

### Compute Infrastructure

#### Hardware

NVIDIA GPU on Kaggle Notebooks (CUDA device).

#### Software

- Python 3
- `transformers` (Hugging Face)
- `torch` (PyTorch)
- `scikit-learn`
- `wandb` for experiment tracking
- `huggingface_hub` for model deployment

## Citation

**BibTeX:**

```bibtex
@misc{laishram2025distilbertgoodreads,
  author       = {Laishram Khumanleima Chanu},
  title        = {distilbert-goodreads-genres: DistilBERT fine-tuned for Goodreads genre classification},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/LKHUMANLEIMA/distilbert-goodreads-genres}},
  note         = {MLOps assignment, IIT Jodhpur}
}
```

**APA:**

[Model]. Hugging Face. https://huggingface.co/LKHUMANLEIMA/distilbert-goodreads-genres
## Glossary

- **DistilBERT** — A smaller, faster, distilled version of BERT with roughly 40% fewer parameters while retaining ~97% of BERT's language-understanding performance.
- **Weighted F1** — A class-frequency-weighted harmonic mean of precision and recall, useful when class support varies.
- **Fine-tuning** — Continuing the training of a pre-trained model on a task-specific dataset to adapt it to a downstream task.

## More Information

Source notebook and training scripts were run on Kaggle. Experiment tracking was done via Weights & Biases under the project `distilbert-goodreads-genres`.

## Model Card Authors

Laishram Khumanleima Chanu (g25ait2056), IIT Jodhpur.

## Model Card Contact

Please open an issue or discussion on the [model repository](https://huggingface.co/LKHUMANLEIMA/distilbert-goodreads-genres) on the Hugging Face Hub.

