# Research Project
Leveraging Large Language Models for Classifying Deliberative Elements in Public Discourse.

## General Setup
1. Clone Repository
2. Setup Python Version: 3.12.0
3. Run "pip install requirements.txt"

## Sentiment Analysis
1. Add the datasets for each annotator in the `data` folder.
2. Modify the file paths in `main_hard.py` and `main_soft.py` to point to the correct CSV files.
3. Run `main_hard.py` and `main_soft.py` to run all the methods for sentiment analysis.
4. The results will be saved in the `sentiment` folder.
5. Modify the file paths in `create_hard_and_soft_labels.py`, `iaa_calculations.py`, and `evaluation.py` to point to the results and datasets.
6. Run `create_hard_and_soft_labels.py` to create the "real" hard and soft labels for the annotators.
7. Run `iaa_calculations.py` to calculate the inter-annotator agreement.
8. Run `evaluation.py` to calculate the evaluation metrics and to generate plots.