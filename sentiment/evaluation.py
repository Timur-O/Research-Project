import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def accuracy(true_values, predicted_values):
    """
    Calculate the accuracy of the model (hard scenario)

    :param true_values: The true values
    :param predicted_values: The predicted values
    :return: The percentage accuracy
    """
    return np.sum(true_values == predicted_values) / len(true_values)


def precision(true_values, predicted_values):
    """
    Calculate the precision of the model (hard scenario)

    :param true_values: The true values
    :param predicted_values: The predicted values
    :return: The precision value
    """
    tp = np.sum((predicted_values == true_values) & (predicted_values > 0))
    fp = np.sum((predicted_values != true_values) & (predicted_values > 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(true_values, predicted_values):
    """
    Calculate the recall of the model (hard scenario)

    :param true_values: The true values
    :param predicted_values: The predicted values
    :return: The recall value
    """
    tp = np.sum((predicted_values == true_values) & (true_values > 0))
    fn = np.sum((predicted_values != true_values) & (true_values > 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(true_values, predicted_values):
    """
    Calculate the F1-Score of the model (hard scenario)

    :param true_values: The true values
    :param predicted_values: The predicted values
    :return: The F1-Score
    """
    prec = precision(true_values, predicted_values)
    rec = recall(true_values, predicted_values)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two vectors.

    :param a: Vector A
    :param b: Vector B
    :return: The Cosine-Similarity value
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def fuzzy_accuracy(true_values, predicted_values):
    """
    Calculate the accuracy of the model (soft scenario)

    :param true_values: The true values
    :param predicted_values: The predicted values
    :return: The percentage accuracy
    """
    similarities = [cosine_similarity(true_values[i], predicted_values[i]) for i in range(len(true_values))]
    return np.mean(similarities)


def fuzzy_precision(true_values, predicted_values):
    """
    Calculate the precision of the model (soft scenario)

    :param true_values: The true values
    :param predicted_values: The predicted values
    :return: The precision value
    """
    return np.sum(np.minimum(true_values, predicted_values)) / np.sum(predicted_values) if np.sum(predicted_values) > 0 else 0


def fuzzy_recall(true_values, predicted_values):
    """
    Calculate the recall of the model (soft scenario)

    :param true_values: The true values
    :param predicted_values: The predicted values
    :return: The recall value
    """
    return np.sum(np.minimum(true_values, predicted_values)) / np.sum(true_values) if np.sum(true_values) > 0 else 0


def fuzzy_f1_score(true_values, predicted_values):
    """
    Calculate the F1-Score of the model (soft scenario)

    :param true_values: The true values
    :param predicted_values: The predicted values
    :return: The F1-Score
    """
    prec = fuzzy_precision(true_values, predicted_values)
    rec = fuzzy_recall(true_values, predicted_values)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


def load_values(filename, column_index):
    """
    Load the values from the file

    :param filename: The filename
    :param column_index: The column index to start from
    :return: The (numpy) array with the values
    """
    return pd.read_csv(filename, encoding='latin1').iloc[:, column_index:].values


if __name__ == "__main__":
    """
    The main function to evaluate the performance of the models once the results have been generated.
    """
    # Load the true and predicted values
    hard_true = np.array(load_values('../../Results/hard_labels.csv', 1))
    soft_true = np.array(load_values('../../Results/soft_labels.csv', 1))

    pred_zero_hard = np.array(load_values('../../Results/zero_shot_res_hard.csv', 0))
    pred_zero_soft = np.array(load_values('../../Results/zero_shot_res_soft.csv', 0))

    pred_few_hard = np.array(load_values('../../Results/few_shot_res_hard.csv', 0))
    pred_few_soft = np.array(load_values('../../Results/few_shot_res_soft.csv', 0))

    pred_cot_zero_hard = np.array(load_values('../../Results/cot_zero_res_hard.csv', 0))
    pred_cot_zero_soft = np.array(load_values('../../Results/cot_zero_res_soft.csv', 0))

    pred_cot_few_hard = np.array(load_values('../../Results/cot_few_res_hard.csv', 0))
    pred_cot_few_soft = np.array(load_values('../../Results/cot_few_res_soft.csv', 0))

    # Calculate the evaluation metrics for the Zero-Shot method
    print('Zero Shot Evaluation')
    accuracy_value_zero = accuracy(hard_true, pred_zero_hard)
    precision_value_zero = precision(hard_true, pred_zero_hard)
    recall_value_zero = recall(hard_true, pred_zero_hard)
    f1_value_zero = f1_score(hard_true, pred_zero_hard)
    fuzzy_accuracy_value_zero = fuzzy_accuracy(soft_true, pred_zero_soft)
    fuzzy_precision_value_zero = fuzzy_precision(soft_true, pred_zero_soft)
    fuzzy_recall_value_zero = fuzzy_recall(soft_true, pred_zero_soft)
    fuzzy_f1_value_zero = fuzzy_f1_score(soft_true, pred_zero_soft)

    # Calculate the evaluation metrics for the Few-Shot method
    print('Few Shot Evaluation')
    accuracy_value_few = accuracy(hard_true, pred_few_hard)
    precision_value_few = precision(hard_true, pred_few_hard)
    recall_value_few = recall(hard_true, pred_few_hard)
    f1_value_few = f1_score(hard_true, pred_few_hard)
    fuzzy_accuracy_value_few = fuzzy_accuracy(soft_true, pred_few_soft)
    fuzzy_precision_value_few = fuzzy_precision(soft_true, pred_few_soft)
    fuzzy_recall_value_few = fuzzy_recall(soft_true, pred_few_soft)
    fuzzy_f1_value_few = fuzzy_f1_score(soft_true, pred_few_soft)

    # Calculate the evaluation metrics for the CoT Zero-Shot method
    print('Cot Zero Shot Evaluation')
    accuracy_value_cot_zero = accuracy(hard_true, pred_cot_zero_hard)
    precision_value_cot_zero = precision(hard_true, pred_cot_zero_hard)
    recall_value_cot_zero = recall(hard_true, pred_cot_zero_hard)
    f1_value_cot_zero = f1_score(hard_true, pred_cot_zero_hard)
    fuzzy_accuracy_value_cot_zero = fuzzy_accuracy(soft_true, pred_cot_zero_soft)
    fuzzy_precision_value_cot_zero = fuzzy_precision(soft_true, pred_cot_zero_soft)
    fuzzy_recall_value_cot_zero = fuzzy_recall(soft_true, pred_cot_zero_soft)
    fuzzy_f1_value_cot_zero = fuzzy_f1_score(soft_true, pred_cot_zero_soft)

    # Calculate the evaluation metrics for the CoT Few-Shot method
    print('Cot Few Shot Evaluation')
    accuracy_value_cot_few = accuracy(hard_true, pred_cot_few_hard)
    precision_value_cot_few = precision(hard_true, pred_cot_few_hard)
    recall_value_cot_few = recall(hard_true, pred_cot_few_hard)
    f1_value_cot_few = f1_score(hard_true, pred_cot_few_hard)
    fuzzy_accuracy_value_cot_few = fuzzy_accuracy(soft_true, pred_cot_few_soft)
    fuzzy_precision_value_cot_few = fuzzy_precision(soft_true, pred_cot_few_soft)
    fuzzy_recall_value_cot_few = fuzzy_recall(soft_true, pred_cot_few_soft)
    fuzzy_f1_value_cot_few = fuzzy_f1_score(soft_true, pred_cot_few_soft)

    # Define labels and values
    labels = ['Zero-Shot', 'Few-Shot', 'CoT-Zero', 'CoT-Few']
    x = np.arange(len(labels))

    # Define the values to enter into the charts
    values_accuracy_hard = [accuracy_value_zero, accuracy_value_few, accuracy_value_cot_zero, accuracy_value_cot_few]
    values_accuracy_soft = [fuzzy_accuracy_value_zero, fuzzy_accuracy_value_few, fuzzy_accuracy_value_cot_zero, fuzzy_accuracy_value_cot_few]
    values_precision_hard = [precision_value_zero, precision_value_few, precision_value_cot_zero, precision_value_cot_few]
    values_precision_soft = [fuzzy_precision_value_zero, fuzzy_precision_value_few, fuzzy_precision_value_cot_zero, fuzzy_precision_value_cot_few]
    values_recall_hard = [recall_value_zero, recall_value_few, recall_value_cot_zero, recall_value_cot_few]
    values_recall_soft = [fuzzy_recall_value_zero, fuzzy_recall_value_few, fuzzy_recall_value_cot_zero, fuzzy_recall_value_cot_few]
    values_f1_hard = [f1_value_zero, f1_value_few, f1_value_cot_zero, f1_value_cot_few]
    values_f1_soft = [fuzzy_f1_value_zero, fuzzy_f1_value_few, fuzzy_f1_value_cot_zero, fuzzy_f1_value_cot_few]

    # Create Accuracy Chart
    print(values_accuracy_hard)
    print(values_accuracy_soft)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, values_accuracy_hard, 0.4, label='Hard Accuracy', color='b')
    ax.bar(x + 0.2, values_accuracy_soft, 0.4, label='Soft Accuracy', color='r')
    ax.set_title('Comparison of Accuracy')
    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy')
    ax.set_xticklabels(labels)
    ax.set_xticks(x)
    ax.legend()
    plt.show()

    # Create Precision Chart
    print(values_precision_hard)
    print(values_precision_soft)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, values_precision_hard, 0.4, label='Hard Precision', color='b')
    ax.bar(x + 0.2, values_precision_soft, 0.4, label='Soft Precision', color='r')
    ax.set_title('Comparison of Precision')
    ax.set_xlabel('Method')
    ax.set_ylabel('Precision')
    ax.set_xticklabels(labels)
    ax.set_xticks(x)
    ax.legend()
    plt.show()

    # Create Recall Chart
    print(values_recall_hard)
    print(values_recall_soft)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, values_recall_hard, 0.4, label='Hard Recall', color='b')
    ax.bar(x + 0.2, values_recall_soft, 0.4, label='Soft Recall', color='r')
    ax.set_title('Comparison of Recall')
    ax.set_xlabel('Method')
    ax.set_ylabel('Recall')
    ax.set_xticklabels(labels)
    ax.set_xticks(x)
    ax.legend()
    plt.show()

    # Create F1 Score Chart
    print(values_f1_hard)
    print(values_f1_soft)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, values_f1_hard, 0.4, label='Hard F1 Score', color='b')
    ax.bar(x + 0.2, values_f1_soft, 0.4, label='Soft F1 Score', color='r')
    ax.set_title('Comparison of F1 Scores')
    ax.set_xlabel('Method')
    ax.set_ylabel('F1 Score')
    ax.set_xticklabels(labels)
    ax.set_xticks(x)
    ax.legend()
    plt.show()
