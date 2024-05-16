import pandas as pd
import csv
from sklearn.model_selection import train_test_split

import OllamaCached


def read_data_and_create_soft_labels():
    data_timur = pd.read_csv("../data/timur.csv")
    data_adina = pd.read_csv("../data/adina.csv")
    data_bente = pd.read_csv("../data/bente.csv")
    data_ana = pd.read_csv("../data/ana.csv")
    data_joosje = pd.read_csv("../data/joosje.csv")
    all_data = [data_timur, data_adina, data_bente, data_ana, data_joosje]

    soft_labelled_data = []

    for i in range(0, 25):  # range max should be the amount of annotated labels
        temp = [data_timur.iloc[i].values[5], 0, 0, 0, 0, 0]
        for data in all_data:
            temp[int(data.iloc[i].values[6] + 1)] += 1
        temp[1] /= 5
        temp[2] /= 5
        temp[3] /= 5
        temp[4] /= 5
        temp[5] /= 5
        soft_labelled_data.append(temp)

    return pd.DataFrame(
        soft_labelled_data,
        None,
        ["Datapoint", "Strongly Negative", "Slightly Negative", "Neutral", "Slightly Positive", "Strongly Positive"]
    )


def zero_shot(model_name, input_row):
    system_text = ("You are a sentiment analysis model. Analyze the given text and provide a probability distribution "
                   "across five sentiment categories: Strongly Negative, Slightly Negative, Neutral, Slightly Positive,"
                   " and Strongly Positive. Return your results as a Python list of floats, where each element "
                   "represents the probability of the corresponding sentiment category. The sum of all probabilities "
                   "must equal 1.0.")
    prompt = "Textual Input: " + input_row
    return OllamaCached.zero_shot(model_name, system_text, prompt)


def few_shot(model_name, training_data, input_row):
    system_text = ("You are a sentiment analysis model. Analyze the given text and provide a probability distribution "
                   "across five sentiment categories: Strongly Negative, Slightly Negative, Neutral, Slightly Positive,"
                   " and Strongly Positive. Return your results as a Python list of floats, where each element "
                   "represents the probability of the corresponding sentiment category. The sum of all probabilities "
                   "must equal 1.0.")
    prompt = "Textual Input: " + input_row

    formatted_training = []
    for training_piece in training_data:
        formatted_training.append([training_piece[0], str(training_piece[1:6])])

    return OllamaCached.few_shot(model_name, system_text, formatted_training, prompt)


def chain_of_thought_zero(model_name, input_row):
    system_text = ("You are a sentiment analysis model. Analyze the given text and provide a probability distribution "
                   "across five sentiment categories: Strongly Negative, Slightly Negative, Neutral, Slightly Positive,"
                   " and Strongly Positive. Return your results as a Python list of floats, where each element "
                   "represents the probability of the corresponding sentiment category. The sum of all probabilities "
                   "must equal 1.0.")
    prompt = "Textual Input: " + input_row
    return OllamaCached.chain_of_reasoning_zero_shot(model_name, system_text, prompt)


def chain_of_thought_few(model_name, training_data, input_row):
    system_text = ("You are a sentiment analysis model. Analyze the given text and provide a probability distribution "
                   "across five sentiment categories: Strongly Negative, Slightly Negative, Neutral, Slightly Positive,"
                   " and Strongly Positive. Return your results as a Python list of floats, where each element "
                   "represents the probability of the corresponding sentiment category. The sum of all probabilities "
                   "must equal 1.0.")
    prompt = "Textual Input: " + input_row

    formatted_training = []
    for training_piece in training_data:
        explanation = OllamaCached.generate_explanation(model_name, training_piece)
        explained_result = explanation + " Thus the final answer is: " + str(training_piece[1:6])
        formatted_training.append([training_piece[0], explained_result])

    return OllamaCached.chain_of_reasoning_few_shot(model_name, system_text, formatted_training, prompt)


def write_to_file(name, data):
    with open(name + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in data:
            csvwriter.writerow(row)


if __name__ == "__main__":
    soft_labels = read_data_and_create_soft_labels()
    model = "llama3"

    zero_shot_results = []
    few_shot_results = []
    chain_of_thought_zero_results = []
    chain_of_thought_few_results = []

    # Split into different sets? (train/validate/test) - 60% training, 20% validation, 20% testing
    train_df, temp_df = train_test_split(soft_labels, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    for i in range(0, len(test_df)):
        input_and_target_results = test_df.iloc[i].values  # 0 = input, 5 next values are target soft labels
        zero_shot_predicted = [0, 0, 0, 0, 0]
        few_shot_predicted = [0, 0, 0, 0, 0]
        chain_of_thought_zero_predicted = [0, 0, 0, 0, 0]
        chain_of_thought_few_predicted = [0, 0, 0, 0, 0]

        for j in range(0, 3):
            zero_shot_predicted += zero_shot(model, input_and_target_results[0])
            few_shot_predicted += few_shot(model, train_df, input_and_target_results[0])
            chain_of_thought_zero_predicted += chain_of_thought_zero(model, input_and_target_results[0])
            chain_of_thought_few_predicted += chain_of_thought_few(model, train_df, input_and_target_results[0])

        zero_shot_predicted /= 3
        zero_shot_results.append(zero_shot_predicted)

        few_shot_predicted /= 3
        few_shot_results.append(few_shot_predicted)

        chain_of_thought_zero_predicted /= 3
        chain_of_thought_zero_results.append(chain_of_thought_zero_predicted)

        chain_of_thought_few_predicted /= 3
        chain_of_thought_few_results.append(chain_of_thought_few_predicted)

    write_to_file("zero_shot_res", zero_shot_results)
    write_to_file("few_shot_res", few_shot_results)
    write_to_file("chain_of_thought_zero", chain_of_thought_zero_results)
    write_to_file("chain_of_thought_few", chain_of_thought_few_results)
