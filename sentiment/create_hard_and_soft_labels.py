import pandas as pd
import csv

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data_timur = pd.read_csv("../data/timur.csv")
    data_adina = pd.read_csv("../data/adina.csv")
    data_bente = pd.read_csv("../data/bente.csv")
    data_ana = pd.read_csv("../data/ana.csv")
    data_joosje = pd.read_csv("../data/joosje.csv")
    all_data = [data_timur, data_adina, data_bente, data_ana, data_joosje]

    hard_labelled_data = []
    soft_labelled_data = []

    for al in range(0, 50):  # range max should be the amount of annotated labels
        temp = [data_timur.iloc[al].values[5], 0, 0, 0, 0, 0]
        for data in all_data:
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

    train_df_hard, temp_df_hard = train_test_split(hard_labelled_data, test_size=0.4, random_state=42)
    val_df_hard, test_df_hard = train_test_split(temp_df_hard, test_size=0.5, random_state=42)

    train_df_soft, temp_df_soft = train_test_split(soft_labelled_data, test_size=0.4, random_state=42)
    val_df_soft, test_df_soft = train_test_split(temp_df_soft, test_size=0.5, random_state=42)

    with open('soft_labels.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in test_df_soft:
            csvwriter.writerow(row)

    with open('hard_labels.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in test_df_hard:
            csvwriter.writerow(row)
