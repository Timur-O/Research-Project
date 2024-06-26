from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import ttest_ind, f_oneway
import pandas as pd
import numpy as np


def calculate_bias(annotations):
    """
    Calculate bias in the annotations using one-way ANOVA.

    :param annotations: The list of annotations, where each annotator is a list of scores.
    :return: The array of average scores and the respective p-value
    """
    # Compute average score for each annotator
    average_scores = [np.mean(annotator) for annotator in annotations]

    # Perform t-test for bias detection
    f_stat, p_value = f_oneway(*annotations)

    return average_scores, p_value


def split_half_reliability(annotations, num_splits=10000):
    """
    Calculate split-half reliability using Fleiss' Kappa.

    :param annotations: The annotations for which to calculate the mean Kappa
    :param num_splits: The number of times to split the data
    :return: The mean Fleiss' Kappa and the p-value of the t-test
    """
    kappa_values = []

    # Perform split-half reliability calculation num_splits times
    for _ in range(num_splits):
        # Randomly split the dataset into two halves
        split_indices = np.random.choice(len(annotations[0]), size=len(annotations[0]) // 2, replace=False)
        split_1 = np.transpose([annotator[split_indices] for annotator in annotations])
        split_2 = np.transpose([np.delete(annotator, split_indices) for annotator in annotations])

        # Calculate Fleiss' Kappa for each split
        kappa_split_1 = fleiss_kappa(split_1)
        kappa_split_2 = fleiss_kappa(split_2)

        # Save the mean of the two Fleiss' Kappa values
        kappa_values.append((kappa_split_1 + kappa_split_2) / 2)

    # Calculate mean Fleiss' Kappa
    mean_kappa = np.mean(kappa_values)

    # Perform t-test for split-half reliability
    _, p_value = ttest_ind(kappa_values, [mean_kappa] * num_splits)

    # Return the mean Fleiss' Kappa and the p-value
    return mean_kappa, p_value


if __name__ == "__main__":
    """
    Calculate the bias and split-half reliability for the annotations of the annotators.
    """
    # Load the data, drop NaN values and extract the annotations
    data_timur = pd.read_csv("../data/timur.csv").iloc[:, 6].dropna().values
    data_adina = pd.read_csv("../data/adina.csv").iloc[:, 6].dropna().values
    data_bente = pd.read_csv("../data/bente.csv").iloc[:, 6].dropna().values
    data_ana = pd.read_csv("../data/ana.csv").iloc[:, 6].dropna().values
    data_joosje = pd.read_csv("../data/joosje.csv").iloc[:, 6].dropna().values

    # Combine the annotations into one list
    annotations = [
        data_timur,
        data_adina,
        data_bente,
        data_ana,
        data_joosje
    ]

    # Calculate bias and print it
    bias_p_value = calculate_bias(annotations)
    print("Bias: average ratings and p-value (one-way ANOVA):", bias_p_value)

    # Calculate split-half reliability and print it
    split_half_p_value = split_half_reliability(annotations)
    print("Split-half reliability mean kappa and p-value:", split_half_p_value)
