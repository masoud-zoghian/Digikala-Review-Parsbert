# 🛒 Digikala Customer Review Analysis and Classification with ParsBERT

![Language](https://img.shields.io/badge/Language-Persian-green)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-blue)
![Multi-task](https://img.shields.io/badge/Multi--task-Learning-orange)
![Semi-supervised](https://img.shields.io/badge/Semi--supervised-Learning-purple)


---

## 📑 Table of Contents

1. [Project Overview](#1--project-overview)
2. [Dataset](#2--dataset)
3. [Data Preprocessing](#3--data-preprocessing)
4. [Baseline Model: TF-IDF + Logistic Regression](#4--baseline-model-tf-idf--logistic-regression)
5. [Main Model: ParsBERT Multi-task Learning](#5--main-model-parsbert-multi-task-learning)
6. [ParsBERT Evaluation](#6--parsbert-evaluation)
7. [Error Analysis](#7--error-analysis)
8. [Semi-supervised Learning](#8--semi-supervised-learning)
9. [Retraining with Semi-supervised Data](#9--retraining-with-semi-supervised-data)
10. [Memory and Storage Management](#10--memory-and-storage-management)
11. [Tools and Libraries](#11--tools-and-libraries)
12. [Suggested Project Structure](#12--suggested-project-structure)
13. [Main Challenges](#13--main-challenges)
14. [Next Steps](#14--next-steps)
15. [Conclusion](#15--conclusion)

---

## 1. 🧩 Project Overview

### 📌 Project Definition

The goal of this project is to design and train a Natural Language Processing (NLP) system that automatically analyzes Digikala customer reviews. Using the review text, the system tries to identify users' opinions about products and classify the overall review score into several levels.

The project uses the Persian language model **ParsBERT**. To benchmark the language model, a baseline built on **TF-IDF and Logistic Regression** is also implemented.

The project consists of two classification problems, learned simultaneously by a single Multi-task Learning model.

### 🎯 Main Objectives

1. 👍 Detect whether the user recommends the product.
2. ⭐ Classify the rating given to the product.
3. ⚖️ Compare classical machine learning models with a ParsBERT-based model.
4. 🧪 Examine the effect of Semi-supervised Learning on model performance.
5. 📉 Study the challenges of imbalanced data and incomplete labels.

### 🏗️ Model Tasks

#### Task 1: Recommendation Classification

The model must determine the user's opinion about recommending the product to others.

| Class | Description |
|:---|:---|
| ✅ `recommended` | The user recommends the product. |
| ❌ `not_recommended` | The user does not recommend the product. |
| ❓ `no_idea` | The user has no clear opinion about recommending the product. |

#### Task 2: Rating Classification

Numerical user ratings are converted into three ranges:

| Class | Score Range |
|:---|:---|
| 🔻 `low` | 0 to less than 2.5 |
| 🔸 `medium` | 2.5 to less than 4 |
| 🔺 `high` | 4 to 5 |

Thus, instead of predicting a numerical rating, the model predicts the rating level as one of the three classes above.

---

## 2. 📂 Dataset

The project data consists of Digikala customer reviews.

### 📊 Data Size

| Split | Records |
|:---|---:|
| Train | 4,308,541 |
| Validation | 922,757 |
| Test | 923,673 |
| **Total** | **6,154,971** |

> 🧹 During duplicate removal, **1,318** exact duplicate records were dropped.

### 🛍️ Number of Products

| Split | Products |
|:---|---:|
| Train | 232,119 |
| Validation | 49,739 |
| Test | 49,741 |

### ⚠️ Important Note on Data Splitting

The data was split by `product_id`, not randomly at the row level.

This decision was made to prevent **Data Leakage**: if reviews of the same product appeared in both training and validation data, the model could learn product-specific information during training, making the evaluation overly optimistic.

**Verification results:**

| Check | Result |
|:---|:---:|
| Products shared between Train and Validation | 0 ✅ |
| Products shared between Train and Test | 0 ✅ |

---

## 3. 🧹 Data Preprocessing

Two versions of the review text were created.

### 📝 Version 1: `text_v1`

- Strip leading and trailing whitespace
- Convert newline and tab characters to spaces
- Collapse multiple consecutive spaces into one

### 📝 Version 2: `text_v2`

In addition to the version 1 steps, Arabic and Persian characters were unified:

- Convert `ي` to `ی`
- Convert `ك` to `ک`

**Data inspection results:**

| Item | Count |
|:---|---:|
| Arabic character `ي` | 48,709 |
| Arabic character `ك` | 25,357 |
| Approximate number of modified rows | 50,156 |

### 💾 Saving the Processed Data

Data is stored in **Parquet** format:

```text
data/
└── processed/
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```

Parquet was chosen for its good compression and fast read/write speed on large datasets.

### 🏷️ Handling Missing Labels

In the training data, **626,370** rows had no `recommendation_status` label.

These rows must not be mapped to the `no_idea` class, because having no label is not the same as having a neutral opinion.

The following value is used to handle this:

```python
IGNORE_INDEX = -100
```

As a result, unlabeled samples are ignored when computing the loss for the recommendation task.

---

## 4. 🤖 Baseline Model: TF-IDF + Logistic Regression

Before using the ParsBERT language model, a baseline was implemented for comparison.

### 🏛️ Model Architecture

```text
Review Text
     |
     v
Text Preprocessing
     |
     v
TF-IDF Vectorization
     |
     v
Logistic Regression
     |
     v
Classification
```

### ⚙️ TF-IDF Settings

| Parameter | Value |
|:---|:---|
| Analyzer | word |
| N-gram Range | (1, 2) |
| Max Features | 100,000 |
| Min DF | 3 |
| Max DF | 0.95 |
| Data Type | float32 |
| Batch Size | 100,000 |

### ⚙️ Logistic Regression Settings

| Parameter | Value |
|:---|:---|
| Solver | saga |
| Class Weight | balanced |
| Max Iterations | 100 |
| Tolerance | 0.001 |

### 🧪 Baseline Experiment Results

#### A) Recommendation Classification

| C | Accuracy | Macro F1 |
|:---:|:---:|:---:|
| 0.1 | 0.7269 | 0.6147 |
| 1 | 0.7570 | 0.5531 |
| **10** | **0.8142** | **0.6367** |

🏆 The best result for this task was obtained with `C=10` using `text_v1`.

#### B) Rating Classification

| C | Accuracy | Macro F1 |
|:---:|:---:|:---:|
| **0.1** | **0.6513** | **0.5668** |
| 1 | 0.6500 | 0.5638 |
| 10 | 0.6428 | 0.5552 |

🏆 The best result for this task was obtained with `C=0.1` using `text_v2`.

---

## 5. 🧠 Main Model: ParsBERT Multi-task Learning

In the next stage, the Persian language model ParsBERT was used.

**Base model:** `HooshvareLab/bert-fa-base-uncased`

This model is designed for Persian language processing and uses a BERT-based Transformer architecture.

### 📐 Model Specifications

| Property | Value |
|:---|---:|
| Vocabulary Size | 100,000 |
| Hidden Size | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Maximum Position Embeddings | 512 |
| Maximum Input Length | 256 |

### 🏗️ Multi-task Architecture

Instead of training two separate models, a single shared model with two outputs is used.

```text
Review Text
     |
     v
ParsBERT (12 Transformer Layers)
     |
     v
Shared Representation
CLS Token + Dropout (0.2)
     |
     +--------------------------+
     |                          |
     v                          v
Recommendation Head        Rating Head
Linear: 768 → 3            Linear: 768 → 3
(3 Classes)                (3 Classes)
```

### 💻 Model Code

```python
class ParsBERTMultiTask(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.2)

        self.recommendation_head = nn.Linear(hidden_size, 3)
        self.rating_head = nn.Linear(hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        rec_logits = self.recommendation_head(pooled_output)
        rating_logits = self.rating_head(pooled_output)

        return rec_logits, rating_logits
```

### ⚙️ Training Settings

| Parameter | Value |
|:---|:---|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Batch Size | 8 |
| Gradient Accumulation | 2 |
| Effective Batch Size | 16 |
| Maximum Sequence Length | 256 |
| Epochs Completed | 1 |
| Precision | FP16 |
| Random Seed | 42 |

> 💡 **Mixed Precision Training** was used to reduce GPU memory usage.

### 📈 Training Results

The model was trained on **100,000** samples from the training data.

| Metric | Value |
|:---|---:|
| Training Samples | 100,000 |
| Optimizer Steps | 6,250 |
| Training Time | about 16.2 minutes |
| Average Recommendation Loss | 0.76409 |
| Average Rating Loss | 1.45787 |
| Average Total Loss | 1.11098 |

✅ One epoch was completed successfully.

---

## 6. 📊 ParsBERT Evaluation

The initial evaluation was run on **20,000** samples from the Validation set.

### 🧾 Overall Results

| Task | Accuracy | Macro F1 |
|:---|:---:|:---:|
| Recommendation | 0.8565 | 0.6639 |
| Rating | 0.7367 | 0.5739 |

### 📋 Recommendation Classification Report

| Class | Precision | Recall | F1-score | Support |
|:---|:---:|:---:|:---:|---:|
| no_idea | 0.45 | 0.39 | 0.41 | 1,914 |
| not_recommended | 0.78 | 0.54 | 0.64 | 1,708 |
| recommended | 0.91 | 0.96 | 0.94 | 13,468 |

### 📋 Rating Classification Report

| Class | Precision | Recall | F1-score | Support |
|:---|:---:|:---:|:---:|---:|
| low | 0.68 | 0.42 | 0.52 | 3,468 |
| medium | 0.52 | 0.27 | 0.36 | 3,526 |
| high | 0.77 | 0.95 | 0.85 | 13,006 |

### 🔢 Recommendation Confusion Matrix

Class order: `no_idea`, `not_recommended`, `recommended`

| Actual \ Predicted | no_idea | not_recommended | recommended |
|:---|---:|---:|---:|
| **no_idea** | 739 | 186 | 989 |
| **not_recommended** | 490 | 930 | 288 |
| **recommended** | 423 | 76 | 12,969 |

### 🔢 Rating Confusion Matrix

Class order: `low`, `medium`, `high`

| Actual \ Predicted | low | medium | high |
|:---|---:|---:|---:|
| **low** | 1,448 | 373 | 1,647 |
| **medium** | 499 | 948 | 2,079 |
| **high** | 177 | 492 | 12,337 |

### 🔍 Results Analysis

- ✅ The model performs very well on the `recommended` class.
- ⚠️ In the Recommendation task, `no_idea` is often confused with `recommended`.
- 🔺 In the Rating task, the `high` class performs best.
- 🔸 `medium` is the hardest class to detect; its recall is only **0.27**.
- ⚖️ Class imbalance is likely an important factor behind these performance gaps.

### ⚖️ Baseline vs. ParsBERT

| Task | Model | Accuracy | Macro F1 |
|:---|:---|:---:|:---:|
| Recommendation | TF-IDF | 0.8142 | 0.6367 |
| Recommendation | **ParsBERT** | **0.8565** | **0.6639** |
| Rating | TF-IDF | 0.6513 | 0.5668 |
| Rating | **ParsBERT** | **0.7367** | **0.5739** |

🏆 ParsBERT achieves a higher Macro F1 on both tasks.

> ⚠️ **Important note on the comparison:** The baseline was evaluated on the entire Validation set, while ParsBERT was evaluated on a 20,000-sample subset of it. This comparison is therefore preliminary and indicative; for a definitive conclusion, both models must be evaluated on the same evaluation set.

---

## 7. 🔬 Error Analysis

This stage aims to examine the samples the model predicted incorrectly.

**Completed steps:**

- Computing Accuracy and Macro F1
- Extracting the Classification Report
- Computing the Confusion Matrix
- Saving predictions for later analysis

**⚠️ Technical issue:**

Because the sample order differed between shuffling and saving predictions, predictions could not be reliably matched to the original review text. Text-based error analysis is therefore not yet complete.

**🛠️ Planned fix:** Re-run inference on the Validation data while preserving each review's `id`, or without shuffling the data.

---

## 8. 🧪 Semi-supervised Learning

Since some reviews lack a Recommendation label, semi-supervised learning is used to exploit the unlabeled data.

In this approach, the trained ParsBERT model is run on the unlabeled data, and **pseudo labels** are generated for samples where the model is sufficiently confident.

### 🪜 Steps

1. Extract the 626,370 unlabeled reviews.
2. Randomly select 50,000 samples with seed 42.
3. Predict with ParsBERT.
4. Compute the class probabilities for each sample.
5. Apply confidence thresholds to select samples.
6. Remove duplicates and run label quality control.

### 📊 Initial Prediction Distribution

| Class | Count |
|:---|---:|
| recommended | 43,370 |
| no_idea | 3,966 |
| not_recommended | 2,664 |
| **Total** | **50,000** |

The average prediction confidence was **0.8786**.

### 🎚️ Confidence Threshold Analysis

A general threshold of 0.90 was tried first, but it caused most selected samples to belong to the `recommended` class, leaving very few samples from the minority classes.

For this reason, **different thresholds per class** were used.

### ⚙️ Final Pseudo Labeling Settings

| Class | Confidence Threshold | Final Count |
|:---|:---:|---:|
| not_recommended | 0.70 | 1,033 |
| recommended | 0.90 | 3,099 |
| no_idea | Excluded from pseudo labels | 0 |
| **Total** | — | **4,132** |

To keep the majority class from dominating, the number of `recommended` samples was capped at **three times** the number of minority-class samples.

### ✅ Pseudo Label Quality Control

| Check | Result |
|:---|:---:|
| Final number of samples | 4,132 |
| Duplicate IDs | 0 |
| Missing text | 0 |
| Overlap with labeled data | 0 |
| Probability sum error | 0 |
| Rating label | -100 |

Since only the Recommendation label was generated for these samples, their Rating loss is ignored during training.

---

## 9. 🏋️ Retraining with Semi-supervised Data

After generating the pseudo labels, a combined dataset was created.

It was ultimately decided to use **10,000** original samples plus the **4,132** pseudo-labeled samples, instead of the 100,000 original samples.

### 🧩 Training Data Composition

| Data Type | Count |
|:---|---:|
| Original labeled data | 10,000 |
| Pseudo-labeled data | 4,132 |
| **Total** | **14,132** |

### ⚖️ Weighting the Pseudo Samples

To reduce the effect of possible pseudo-label errors, the loss weight of pseudo samples is lower than that of real samples.

| Sample Type | Loss Weight |
|:---|:---:|
| Original Labeled Data | 1.0 |
| Pseudo Labeled Data | 0.3 |

### ⚙️ Training Settings

| Parameter | Value |
|:---|:---|
| Batch Size | 8 |
| Gradient Accumulation | 2 |
| Effective Batch Size | 16 |
| Learning Rate | 1e-5 |
| Weight Decay | 0.01 |
| Pseudo Loss Weight | 0.3 |
| Epochs | 1 |
| Total Batches | 1,767 |
| Optimizer Steps (approx.) | 884 |

### 🚧 Status: Semi-supervised Training Needs Checkpoint Verification

#### 💾 Checkpoint Status

During training, checkpoint-save messages appeared in the run output at several batches, including 400, 500, 600, and 700.

However, when the saved file was re-inspected, the latest verified state corresponded to **batch 300**.

This discrepancy shows that a "saved" message in the run output alone is not sufficient proof that the latest state was permanently written to Google Drive.

> ❗ The exact cause is still unknown and must be determined by checking the file path, the saving operation, Drive synchronization, and the actual contents of the checkpoint.

---

## 10. 💾 Memory and Storage Management

Because of the large data volume and the size of ParsBERT, storage management has been one of the project's major challenges.

### 📦 Storage Status

| Item | Value |
|:---|:---|
| Total Google Drive capacity | about 15GB |
| Reported free space | about 22% (≈ 3.3GB) |
| Initial model checkpoint size | about 0.61GB |
| Semi-supervised training checkpoint size | about 1.82GB |
| Some processed data and TF-IDF matrices | a few hundred MB |
| Total project file size (in one check) | about 4.01GB |

### ✅ Actions Taken

- 🗑️ Removed unneeded Hugging Face cache after saving the model and tokenizer
- 📦 Stored data in Parquet format
- 💾 Saved training checkpoints
- 📁 Separated model, data, report, and output folders

### ⚠️ Key Challenge

Safely saving a checkpoint usually requires temporary space as well. So when saving a 1.82GB file, more space than the final file size may be needed.

---

## 11. 🖥️ Tools and Libraries

| Tool | Purpose |
|:---|:---|
| 🐍 Python | Main project language |
| 🔢 NumPy | Array processing |
| 🐼 Pandas | Data management |
| 📊 Scikit-learn | TF-IDF, Logistic Regression, and evaluation metrics |
| 🔥 PyTorch | Implementing and training the neural network |
| 🤗 Transformers | Loading ParsBERT and the tokenizer |
| 🤗 Hugging Face | Language model and tokenizer |
| 🗃️ PyArrow / Parquet | Data storage |
| ☁️ Google Colab | Experiment environment |
| 💽 Google Drive | Storing data and checkpoints |

---

## 12. 📁 Suggested Project Structure

```text
Digikala_Project/
│
├── data/
│   └── processed/
│       ├── train.parquet
│       ├── validation.parquet
│       └── test.parquet
│
├── notebooks/
│   ├── 03_transfer_learning_finetuning.ipynb
│   ├── 04_error_analysis.ipynb
│   ├── 05_semi_supervised.ipynb
│   └── 06_semi_supervised_training.ipynb
│
├── 03_transformer_parsbert/
│   ├── checkpoints/
│   │   └── parsbert_multitask_epoch1.pt
│   ├── tokenizer/
│   ├── models/
│   ├── metrics/
│   └── predictions/
│
├── 04_semi_supervised/
│   ├── checkpoints/
│   │   └── semi_supervised_14k_resume.pt
│   ├── models/
│   ├── metrics/
│   ├── logs/
│   └── cache/
│
└── README.md
```

> ℹ️ This structure gives an overview of the main files and paths used in the project; depending on the final state of the workspace, some files may be located elsewhere.

---

## 13. ⚠️ Main Challenges

| Challenge | Description | Status |
|:---|:---|:---:|
| Imbalanced data | Differences in the number of samples per class | 🔍 Investigated |
| Incomplete labels | 626,370 missing Recommendation labels | ✅ Handled |
| Data Leakage | Possible product overlap between splits | ✅ Prevented |
| GPU memory usage | Training a large language model | 🟡 Partially handled |
| Google Drive space | Large checkpoint sizes | 🟠 Needs management |
| Prediction order | Mismatch between IDs and text in error analysis | 🟠 Needs fixing |
| Checkpoint | Discrepancy between reported and saved batch | ❌ Unresolved |
| GPU limitations | Changing GPU and CPU runtime conditions | 🟠 Needs investigation |


---

## 14. 🚀 Next Steps

The following steps are suggested to complete the project:

- [ ] 1. Carefully verify the semi-supervised training checkpoint and make sure it saves reliably.
- [ ] 2. Resume training from the last verified batch.
- [ ] 3. Complete the evaluation of the semi-supervised model on the same Validation set used for the baseline.
- [ ] 4. Compare the main model and the semi-supervised model using Accuracy, Macro F1, and Confusion Matrix.
- [ ] 5. Run Error Analysis while preserving each sample's ID.
- [ ] 6. Run the final evaluation on the Test set, which was not used for model selection.
- [ ] 7. Examine performance on minority classes, especially `no_idea` and `medium`.
- [ ] 8. Save the final model, tokenizer, configuration, and results to a defined path.

---

## 15. 🏁 Conclusion

In this project, a Persian review classification system was designed using the ParsBERT model and Multi-task Learning.

First, a baseline based on TF-IDF and Logistic Regression was built. Then ParsBERT was trained with two independent heads to predict the recommendation status and the rating level.

Initial results showed that ParsBERT achieved a higher Macro F1 than the baseline on both tasks. However, performance on minority classes still has room for improvement.

Next, Semi-supervised Learning was used to take advantage of part of the unlabeled data. After applying confidence filters and quality control, 4,132 pseudo-labeled samples were produced.

Currently, the initial evaluation of the main model is done, but the semi-supervised training, error analysis, and final Test set evaluation still need to be completed and verified.

This project shows that combining Persian language models, multi-task learning, and semi-supervised learning can be a suitable approach for analyzing user reviews at scale.

> 📝 **Note on results:** All numbers reported in this README come from the recorded results of the project's experiments. The semi-supervised training results are not presented as the model's final results, because their final evaluation has not yet been confirmed.
