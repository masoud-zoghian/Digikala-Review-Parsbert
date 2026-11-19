import os
import json

BASE = os.path.join("project-3")

folders = [
    "data",
    "notebooks",
    "artifacts",
    "reports",
]

for folder in folders:
    os.makedirs(os.path.join(BASE, folder), exist_ok=True)

# نوت‌بوک خالی و معتبر (فایل ipynb خالی باز نمی‌شود، باید JSON معتبر باشد)
empty_notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}

notebooks = [
    "01_descriptive_analysis.ipynb",
    "02_baselines.ipynb",
    "03_transfer_learning_finetuning.ipynb",
    "04_semi_supervised.ipynb",
]

for nb in notebooks:
    nb_path = os.path.join(BASE, "notebooks", nb)
    if not os.path.exists(nb_path):
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(empty_notebook, f, indent=1)

root_files = ["requirements.txt", "README.md", ".gitignore"]

for name in root_files:
    file_path = os.path.join(BASE, name)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")

print("The project structure was created:", os.path.abspath(BASE))
