import numpy as np
from scipy.stats import ttest_ind, f_oneway
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa


def calculate_bias(annotations):
    # Compute average score for each annotator
    average_scores = [np.mean(annotator) for annotator in annotations]

    # Perform t-test for bias detection
    f_stat, p_value = f_oneway(*annotations)

    return average_scores, p_value


def split_half_reliability(annotations, num_splits=10000):
    kappa_values = []

    for _ in range(num_splits):
        # Randomly split the dataset into two halves
        split_indices = np.random.choice(len(annotations[0]), size=len(annotations[0]) // 2, replace=False)
        split_1 = np.transpose([annotator[split_indices] for annotator in annotations])
        split_2 = np.transpose([np.delete(annotator, split_indices) for annotator in annotations])

        # Calculate Fleiss' Kappa for each split
        kappa_split_1 = fleiss_kappa(split_1)
        kappa_split_2 = fleiss_kappa(split_2)

        kappa_values.append((kappa_split_1 + kappa_split_2) / 2)

    # Calculate mean Fleiss' Kappa and standard error
    mean_kappa = np.mean(kappa_values)
    std_error = np.std(kappa_values) / np.sqrt(num_splits)

    # Perform t-test for split-half reliability
    _, p_value = ttest_ind(kappa_values, [mean_kappa] * num_splits)

    return mean_kappa, p_value


if __name__ == "__main__":
    data_timur = pd.read_csv("../data/timur.csv").iloc[:, 6].dropna().values
    data_adina = pd.read_csv("../data/adina.csv").iloc[:, 6].dropna().values
    data_bente = pd.read_csv("../data/bente.csv").iloc[:, 6].dropna().values
    data_ana = pd.read_csv("../data/ana.csv").iloc[:, 6].dropna().values
    data_joosje = pd.read_csv("../data/joosje.csv").iloc[:, 6].dropna().values

    annotations = [
        data_timur,
        data_adina,
        data_bente,
        data_ana,
        data_joosje
    ]

    bias_p_value = calculate_bias(annotations)
    print("Bias: average ratings and p-value (one-way ANOVA):", bias_p_value)

    split_half_p_value = split_half_reliability(annotations)
    print("Split-half reliability mean kappa and p-value:", split_half_p_value)
