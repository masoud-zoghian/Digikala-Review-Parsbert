# Libraries Used

This document lists the third-party packages and Python standard-library modules used or specified for this project. Third-party packages must be installed with `pip`; standard-library modules are included with Python.

## Third-Party Libraries

Install these packages using `requirements.txt`.

| Package | Imported Module / API | Used For |
|---|---|---|
| `numpy` | `numpy` | Numerical operations on arrays and matrices. |
| `pandas` | `pandas` | Loading, organizing, transforming, and analyzing tabular data. |
| `scikit-learn` | `sklearn.feature_extraction.text.TfidfVectorizer` | Converting text into numerical TF-IDF feature vectors. |
| `scikit-learn` | `sklearn.linear_model.LogisticRegression` | Training a Logistic Regression classification model. |
| `scikit-learn` | `sklearn.metrics` | Evaluating models with accuracy, F1-score, classification reports, and confusion matrices. |
| `scipy` | `scipy.sparse` | Creating, stacking, loading, and saving sparse matrices; useful for memory-efficient text features. |
| `pyarrow` | `pyarrow.parquet` | Reading data stored in Parquet files. |
| `duckdb` | `duckdb` | Querying and processing structured data with SQL. |
| `joblib` | `joblib` | Saving and loading trained models and vectorizers. |
| `psutil` | `psutil` | Inspecting system and process resources, such as memory and CPU usage. |
| `torch` | `torch`, `torch.nn`, `torch.utils.data`, `torch.optim` | Deep learning tensors, neural-network layers, datasets, data loaders, and optimizers. |
| `transformers` | `transformers.AutoTokenizer`, `transformers.AutoConfig`, `transformers.AutoModel` | Loading pretrained Transformer tokenizers, configurations, and models. |
| `matplotlib` | `matplotlib.pyplot` | Creating plots and visualizing results. |
| `seaborn` | `seaborn` | Statistical data visualization, commonly used with Matplotlib. |
| `tqdm` | `tqdm.auto` | Displaying progress bars during long-running operations. |
| `ipython` | `IPython.display.display` | Displaying rich output in Jupyter/IPython notebooks. |

## Python Standard Library

These modules are included with Python and do not need to be installed separately.

| Module | Used For |
|---|---|
| `pathlib` | `Path` — working with filesystem paths. |
| `os` | Operating-system and filesystem operations. |
| `collections` | `Counter` — counting occurrences of items. |
| `re` | Regular expressions for searching, matching, and cleaning text. |
| `time` | Measuring and controlling time-related operations. |
| `json` | Reading and writing JSON data. |
| `datetime` | Working with dates and times. |
| `warnings` | Controlling or filtering warning messages. |
| `gc` | Explicit garbage collection and memory management. |
| `math` | Mathematical functions and constants. |
| `random` | Generating random values and shuffling items. |

## Installation

To install the listed third-party packages:

```bash
python -m pip install -r requirements.txt
```

## Notes

- The import name and the package name are not always identical. For example, install `scikit-learn` but import it as `sklearn`.
- PyTorch installation may need a platform-specific command if you want a particular CUDA version. Check the official PyTorch installation selector for the right command for your system.
- `numpy`, `pandas`, `joblib`, and `psutil` were included because they were listed in the project's dependency summary, even though they do not appear as direct imports in the code snippet supplied.
- `torchvision` is not included because it does not appear in the supplied imports or dependency summary.
