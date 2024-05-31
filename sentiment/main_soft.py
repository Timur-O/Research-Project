import operator

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

    return pd.DataFrame(
        soft_labelled_data,
        None,
        ["Datapoint", "Strongly Negative", "Slightly Negative", "Neutral", "Slightly Positive", "Strongly Positive"]
    )


def zero_shot(model_name, input_row):
    system_text = ("You are a sentiment analysis model. You will analyze the text given by the user and provide a "
                   "probability distribution across five sentiment categories: Strongly Negative, Slightly Negative, "
                   "Neutral, Slightly Positive, and Strongly Positive. The output should be a Python list of floats "
                   "(e.g., [0.2, 0.3, 0.4, 0.05, 0.05]), where each element represents the probability of the "
                   "corresponding sentiment category. The sum of all probabilities must equal 1.0. Follow these "
                   "rules: 1) Return only the Python list of floats. 2) Ensure the sum of the probabilities equals "
                   "1.0. If not, adjust the values proportionally. 3) Do not provide any additional information or "
                   "text in your response. Example response: [0.2, 0.3, 0.4, 0.05, 0.05]")
    prompt = "Textual Input: " + input_row
    return OllamaCached.zero_shot(model_name, system_text, prompt)


def few_shot(model_name, training_data, input_row):
    system_text = ("You are a sentiment analysis model. You will analyze the text given by the user and provide a "
                   "probability distribution across five sentiment categories: Strongly Negative, Slightly Negative, "
                   "Neutral, Slightly Positive, and Strongly Positive. The output should be a Python list of floats "
                   "(e.g., [0.2, 0.3, 0.4, 0.05, 0.05]), where each element represents the probability of the "
                   "corresponding sentiment category. The sum of all probabilities must equal 1.0. Follow these "
                   "rules: 1) Return only the Python list of floats. 2) Ensure the sum of the probabilities equals "
                   "1.0. If not, adjust the values proportionally. 3) Do not provide any additional information or "
                   "text in your response. Example response: [0.2, 0.3, 0.4, 0.05, 0.05]")
    prompt = "Textual Input: " + input_row

    formatted_training = []
    for t in range(0, len(training_data)):
        training_row = training_data.iloc[t].values  # 0 = input, 5 next values are target soft labels
        correct_result = "[" + ", ".join(str(x) for x in training_row[1:6]) + "]"
        formatted_training.append([training_row[0], correct_result])

    return OllamaCached.few_shot(model_name, system_text, formatted_training, prompt)


def chain_of_thought_zero(model_name, input_row):
    system_text = ("You are a sentiment analysis model. You will analyze the text given by the user and provide two "
                   "outputs: 1) A probability distribution across five sentiment categories: Strongly Negative, "
                   "Slightly Negative, Neutral, Slightly Positive, and Strongly Positive. The output should be a "
                   "Python list of floats (e.g., [0.2, 0.3, 0.4, 0.05, 0.05]), where each element represents the "
                   "probability of the corresponding sentiment category. The sum of all probabilities must equal 1.0. "
                   "Follow these rules: a) Return only the Python list of floats for this part. b) Ensure the sum of "
                   "the probabilities equals 1.0. If not, adjust the values proportionally. 2) An explanation or "
                   "reasoning for the results in a separate paragraph. Do not combine the two outputs. The response "
                   "should be structured as follows: First, a separate explanation paragraph. Then, the Python list of "
                   "floats. Example response: The text contains mixed sentiments with a stronger leaning towards "
                   "neutrality. There are slight negative and positive sentiments detected with the use of phrases "
                   "such as \"inconvenient\", \"annoying\", and \"supporting\", but the overall tone is "
                   "neutral. [0.2, 0.3, 0.4, 0.05, 0.05]")
    prompt = "Textual Input: " + input_row
    return OllamaCached.chain_of_reasoning_zero_shot(model_name, system_text, prompt)


def chain_of_thought_few(model_name, training_data, input_row):
    system_text = ("You are a sentiment analysis model. You will analyze the text given by the user and provide two "
                   "outputs: 1) A probability distribution across five sentiment categories: Strongly Negative, "
                   "Slightly Negative, Neutral, Slightly Positive, and Strongly Positive. The output should be a "
                   "Python list of floats (e.g., [0.2, 0.3, 0.4, 0.05, 0.05]), where each element represents the "
                   "probability of the corresponding sentiment category. The sum of all probabilities must equal 1.0. "
                   "Follow these rules: a) Return only the Python list of floats for this part. b) Ensure the sum of "
                   "the probabilities equals 1.0. If not, adjust the values proportionally. 2) An explanation or "
                   "reasoning for the results in a separate paragraph. Do not combine the two outputs. The response "
                   "should be structured as follows: First, a separate explanation paragraph. Then, the Python list of "
                   "floats. Example response: The text contains mixed sentiments with a stronger leaning towards "
                   "neutrality. There are slight negative and positive sentiments detected with the use of phrases "
                   "such as \"inconvenient\", \"annoying\", and \"supporting\", but the overall tone is "
                   "neutral. [0.2, 0.3, 0.4, 0.05, 0.05]")
    prompt = "Textual Input: " + input_row

    formatted_training = []
    for t in range(0, len(training_data)):
        training_row = training_data.iloc[t].values  # 0 = input, 5 next values are target soft labels
        explanation = get_or_gen_explanation(model_name, training_row)
        correct_result = "[" + ", ".join(str(x) for x in training_row[1:6]) + "]"
        explained_result = explanation + "\n " + correct_result
        formatted_training.append([training_row[0], explained_result])

    return OllamaCached.chain_of_reasoning_few_shot(model_name, system_text, formatted_training, prompt)


def get_or_gen_explanation(model_name, training_row):
    with open('../data/explanations_soft.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    temp_exp = ""
    for row in rows:
        if row[0] == training_row[0]:
            try:
                return row[1]
            except IndexError:
                temp_exp = OllamaCached.generate_explanation_soft(model_name, training_row)
                row.append(temp_exp)
                break

    with open('../data/explanations_soft.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    return temp_exp


def write_to_file(name, data):
    with open(name + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in data:
            csvwriter.writerow(row)


def extract_python_array(result):
    string_arr = result.split("[")[1].split("]")[0].split(", ")
    for s in range(0, len(string_arr)):
        string_arr[s] = float(string_arr[s])
    return string_arr


if __name__ == "__main__":
    soft_labels = read_data_and_create_soft_labels()
    model = "llama3:latest"

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
            zero_result_extracted = ""
            while zero_result_extracted == "":
                try:
                    zero_result = zero_shot(model, input_and_target_results[0])
                    zero_result_extracted = extract_python_array(zero_result)
                except Exception as e:
                    print(e)
                    print("Oopsi, trying zero again!")
            zero_shot_predicted = list(map(operator.add, zero_shot_predicted, zero_result_extracted))

            few_result_extracted = ""
            while few_result_extracted == "":
                try:
                    few_result = few_shot(model, train_df, input_and_target_results[0])
                    few_result_extracted = extract_python_array(few_result)
                except Exception as e:
                    print(e)
                    print("Oopsi, trying few again!")
            few_shot_predicted = list(map(operator.add, few_shot_predicted, few_result_extracted))

            cot_zero_result_extracted = ""
            while cot_zero_result_extracted == "":
                try:
                    cot_zero_result = chain_of_thought_zero(model, input_and_target_results[0])
                    cot_zero_result_extracted = extract_python_array(cot_zero_result)
                except Exception as e:
                    print(e)
                    print("Oopsi, trying cot_zero again!")
            chain_of_thought_zero_predicted = list(
                map(operator.add, chain_of_thought_zero_predicted, cot_zero_result_extracted)
            )

            cot_few_result_extracted = ""
            while cot_few_result_extracted == "":
                try:
                    cot_few_result = chain_of_thought_few(model, train_df, input_and_target_results[0])
                    cot_few_result_extracted = extract_python_array(cot_few_result)
                except Exception as e:
                    print(e)
                    print("Oopsi, trying cot_few again!")
            chain_of_thought_few_predicted += cot_few_result_extracted
            chain_of_thought_few_predicted = list(
                map(operator.add, chain_of_thought_few_predicted, cot_few_result_extracted)
            )

        zero_shot_results.append([float(x) / 3.0 for x in zero_shot_predicted])
        few_shot_results.append([float(x) / 3.0 for x in few_shot_predicted])
        chain_of_thought_zero_results.append([float(x) / 3.0 for x in chain_of_thought_zero_predicted])
        chain_of_thought_few_results.append([float(x) / 3.0 for x in chain_of_thought_few_predicted])

    write_to_file("zero_shot_res_soft", zero_shot_results)
    write_to_file("few_shot_res_soft", few_shot_results)
    write_to_file("cot_zero_res_soft", chain_of_thought_zero_results)
    write_to_file("cot_few_res_soft", chain_of_thought_few_results)
