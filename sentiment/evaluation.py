import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def accuracy(true_values, predicted_values):
    return np.sum(true_values == predicted_values) / len(true_values)


def precision(true_values, predicted_values):
    tp = np.sum((predicted_values == true_values) & (predicted_values > 0))
    fp = np.sum((predicted_values != true_values) & (predicted_values > 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(true_values, predicted_values):
    tp = np.sum((predicted_values == true_values) & (true_values > 0))
    fn = np.sum((predicted_values != true_values) & (true_values > 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(true_values, predicted_values):
    prec = precision(true_values, predicted_values)
    rec = recall(true_values, predicted_values)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def fuzzy_accuracy(true_values, predicted_values):
    similarities = [cosine_similarity(true_values[i], predicted_values[i]) for i in range(len(true_values))]
    return np.mean(similarities)


def fuzzy_precision(true_values, predicted_values):
    return np.sum(np.minimum(true_values, predicted_values)) / np.sum(predicted_values) if np.sum(predicted_values) > 0 else 0


def fuzzy_recall(true_values, predicted_values):
    return np.sum(np.minimum(true_values, predicted_values)) / np.sum(true_values) if np.sum(true_values) > 0 else 0


def fuzzy_f1_score(true_values, predicted_values):
    prec = fuzzy_precision(true_values, predicted_values)
    rec = fuzzy_recall(true_values, predicted_values)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


def load_values(filename, column_index):
    return pd.read_csv(filename, encoding='latin1').iloc[:, column_index:].values


if __name__ == "__main__":
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

    print('Zero Shot Evaluation')
    accuracy_value_zero = accuracy(hard_true, pred_zero_hard)
    precision_value_zero = precision(hard_true, pred_zero_hard)
    recall_value_zero = recall(hard_true, pred_zero_hard)
    f1_value_zero = f1_score(hard_true, pred_zero_hard)
    fuzzy_accuracy_value_zero = fuzzy_accuracy(soft_true, pred_zero_soft)
    fuzzy_precision_value_zero = fuzzy_precision(soft_true, pred_zero_soft)
    fuzzy_recall_value_zero = fuzzy_recall(soft_true, pred_zero_soft)
    fuzzy_f1_value_zero = fuzzy_f1_score(soft_true, pred_zero_soft)

    print('Few Shot Evaluation')
    accuracy_value_few = accuracy(hard_true, pred_few_hard)
    precision_value_few = precision(hard_true, pred_few_hard)
    recall_value_few = recall(hard_true, pred_few_hard)
    f1_value_few = f1_score(hard_true, pred_few_hard)
    fuzzy_accuracy_value_few = fuzzy_accuracy(soft_true, pred_few_soft)
    fuzzy_precision_value_few = fuzzy_precision(soft_true, pred_few_soft)
    fuzzy_recall_value_few = fuzzy_recall(soft_true, pred_few_soft)
    fuzzy_f1_value_few = fuzzy_f1_score(soft_true, pred_few_soft)

    print('Cot Zero Shot Evaluation')
    accuracy_value_cot_zero = accuracy(hard_true, pred_cot_zero_hard)
    precision_value_cot_zero = precision(hard_true, pred_cot_zero_hard)
    recall_value_cot_zero = recall(hard_true, pred_cot_zero_hard)
    f1_value_cot_zero = f1_score(hard_true, pred_cot_zero_hard)
    fuzzy_accuracy_value_cot_zero = fuzzy_accuracy(soft_true, pred_cot_zero_soft)
    fuzzy_precision_value_cot_zero = fuzzy_precision(soft_true, pred_cot_zero_soft)
    fuzzy_recall_value_cot_zero = fuzzy_recall(soft_true, pred_cot_zero_soft)
    fuzzy_f1_value_cot_zero = fuzzy_f1_score(soft_true, pred_cot_zero_soft)

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
    x = np.arange(len(labels))  # Create x coordinates for the bars
    values_accuracy_hard = [accuracy_value_zero, accuracy_value_few, accuracy_value_cot_zero, accuracy_value_cot_few]
    values_accuracy_soft = [fuzzy_accuracy_value_zero, fuzzy_accuracy_value_few, fuzzy_accuracy_value_cot_zero, fuzzy_accuracy_value_cot_few]
    values_precision_hard = [precision_value_zero, precision_value_few, precision_value_cot_zero, precision_value_cot_few]
    values_precision_soft = [fuzzy_precision_value_zero, fuzzy_precision_value_few, fuzzy_precision_value_cot_zero, fuzzy_precision_value_cot_few]
    values_recall_hard = [recall_value_zero, recall_value_few, recall_value_cot_zero, recall_value_cot_few]
    values_recall_soft = [fuzzy_recall_value_zero, fuzzy_recall_value_few, fuzzy_recall_value_cot_zero, fuzzy_recall_value_cot_few]
    values_f1_hard = [f1_value_zero, f1_value_few, f1_value_cot_zero, f1_value_cot_few]
    values_f1_soft = [fuzzy_f1_value_zero, fuzzy_f1_value_few, fuzzy_f1_value_cot_zero, fuzzy_f1_value_cot_few]

    # Create Accuracy Chart
    plt.bar(x - 0.2, values_accuracy_hard, 0.4, color='b')
    plt.bar(x + 0.2, values_accuracy_soft, 0.4, color='r')
    plt.title('Comparison of Accuracy')
    plt.xlabel('Method')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, labels)
    plt.show()

    # Create Precision Chart
    plt.bar(x - 0.2, values_precision_hard, 0.4, color='b')
    plt.bar(x + 0.2, values_precision_soft, 0.4, color='r')
    plt.title('Comparison of Precision')
    plt.xlabel('Method')
    plt.ylabel('Precision')
    plt.xticks(x, labels)
    plt.show()

    # Create Recall Chart
    plt.bar(x - 0.2, values_recall_hard, 0.4, color='b')
    plt.bar(x + 0.2, values_recall_soft, 0.4, color='r')
    plt.title('Comparison of Recall')
    plt.xlabel('Method')
    plt.ylabel('Recall')
    plt.xticks(x, labels)
    plt.show()

    # Create F1 Score Chart
    plt.bar(x - 0.2, values_f1_hard, 0.4, color='b')
    plt.bar(x + 0.2, values_f1_soft, 0.4, color='r')
    plt.title('Comparison of F1-Scores')
    plt.xlabel('Method')
    plt.ylabel('F1 Score')
    plt.xticks(x, labels)
    plt.show()
