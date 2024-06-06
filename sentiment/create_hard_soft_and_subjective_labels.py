from sklearn.model_selection import train_test_split
import pandas as pd
import csv


if __name__ == "__main__":
    """
    This script creates hard, soft, and subjective labels for the data. These can be used to evaluate the model.
    """
    # Read the data from the CSV files of the annotators
    data_timur = pd.read_csv("../data/timur.csv")
    data_adina = pd.read_csv("../data/adina.csv")
    data_bente = pd.read_csv("../data/bente.csv")
    data_ana = pd.read_csv("../data/ana.csv")
    data_joosje = pd.read_csv("../data/joosje.csv")
    all_data = [data_timur, data_adina, data_bente, data_ana, data_joosje]

    # Initialize the hard and soft label arrays
    hard_labelled_data = []
    soft_labelled_data = []
    subj_labelled_data = []

    # Go through all the annotated rows and calculate the hard and soft labels
    for al in range(0, 50):  # range max should be the amount of annotated labels
        temp = [data_timur.iloc[al].values[5], 0, 0, 0, 0, 0]
        temp_all_annotators = []
        for data in all_data:
            temp_all_annotators.append(data.iloc[al].values[6])
            temp[int(data.iloc[al].values[6] + 1)] += 1
        temp[1] /= 5
        temp[2] /= 5
        temp[3] /= 5
        temp[4] /= 5
        temp[5] /= 5
        soft_labelled_data.append(temp)
        majority = -1
        majority_percentage = 0
        for t in range(1, 6):
            if temp[t] > majority_percentage:
                majority = t -1
                majority_percentage = temp[t]
        hard_labelled_data.append([temp[0], majority])
        subj_labelled_data.append([temp[0]] + temp_all_annotators)

    # Split the data into train, validation and test sets
    train_df_hard, test_df_hard = train_test_split(hard_labelled_data, test_size=0.3, random_state=42)
    train_df_soft, test_df_soft = train_test_split(soft_labelled_data, test_size=0.3, random_state=42)
    train_df_subj, test_df_subj = train_test_split(subj_labelled_data, test_size=0.3, random_state=42)

    # Write the "correct" labels for soft labels to a CSV file
    with open('soft_labels.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in test_df_soft:
            csvwriter.writerow(row)

    # Write the "correct" labels for hard labels to a CSV file
    with open('hard_labels.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in test_df_hard:
            csvwriter.writerow(row)

    # Write the "correct" labels for subjective labels to a CSV file
    with open('subjective_labels.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in test_df_hard:
            csvwriter.writerow(row)