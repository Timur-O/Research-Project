from sklearn.model_selection import train_test_split
import pandas as pd
import OllamaCached
import csv
import re


def read_data_and_create_hard_labels():
    """
    Reads the data from the CSV files and creates hard labels based on majority.

    :return: A dataframe with the textual inputs and the majority/hard labels.
    """
    # Read data from all the CSV files with the labels of all the annotators
    data_timur = pd.read_csv("../data/timur.csv")
    data_adina = pd.read_csv("../data/adina.csv")
    data_bente = pd.read_csv("../data/bente.csv")
    data_ana = pd.read_csv("../data/ana.csv")
    data_joosje = pd.read_csv("../data/joosje.csv")
    all_data = [data_timur, data_adina, data_bente, data_ana, data_joosje]

    # Create hard labels for each annotated row
    hard_labelled_data = []
    for al in range(0, 50):  # range max should be the amount of annotated rows
        temp = [data_timur.iloc[al].values[5], 0, 0, 0, 0, 0]
        for data in all_data:
            temp[int(data.iloc[al].values[6] + 1)] += 1
        temp[1] /= 5
        temp[2] /= 5
        temp[3] /= 5
        temp[4] /= 5
        temp[5] /= 5
        majority = -1
        majority_percentage = 0
        for t in range(1, 6):
            if temp[t] > majority_percentage:
                majority = t - 1
                majority_percentage = temp[t]
        hard_labelled_data.append([temp[0], majority])

    # Return the dataframe with the correct headers and filtered information
    return pd.DataFrame(
        hard_labelled_data,
        None,
        ["Datapoint", "Sentiment"]
    )


def zero_shot(model_name, input_row):
    """
    Run the zero-shot model on the input row using the provided model.

    :param model_name: The LLM model to use.
    :param input_row: The input row to classify
    :return: The entire response from the model.
    """
    system_text = ("You are a sentiment analysis model. You will analyze the text given by the user and provide the "
                   "sentiment of the text from five sentiment categories: Strongly Negative, Slightly Negative, "
                   "Neutral, Slightly Positive, and Strongly Positive. The output should be a number from zero (0) "
                   "to four (4), which represents the corresponding sentiment category. Follow these rules: 1) Return "
                   "only a number from zero four. 2) Do not provide any additional information or text in your "
                   "response. Example response: 3")
    prompt = "Textual Input: " + input_row
    return OllamaCached.zero_shot(model_name, system_text, prompt)


def few_shot(model_name, training_data, input_row):
    """
    Run the few-shot model on the input row using the provided model.

    :param model_name: The LLM model to use.
    :param training_data: The training data to use for the few-shot model.
    :param input_row: The input row to classify
    :return: The entire response from the model.
    """
    system_text = ("You are a sentiment analysis model. You will analyze the text given by the user and provide the "
                   "sentiment of the text from five sentiment categories: Strongly Negative, Slightly Negative, "
                   "Neutral, Slightly Positive, and Strongly Positive. The output should be a number from zero (0) "
                   "to four (4), which represents the corresponding sentiment category. Follow these rules: 1) Return "
                   "only a number from zero four. 2) Do not provide any additional information or text in your "
                   "response. Example response: 3")
    prompt = "Textual Input: " + input_row

    # Format the training data
    formatted_training = []
    for t in range(0, len(training_data)):
        training_row = training_data.iloc[t].values  # 0 = input, 1 = target hard label
        formatted_training.append([training_row[0], training_row[1]])

    return OllamaCached.few_shot(model_name, system_text, formatted_training, prompt)


def chain_of_thought_zero(model_name, input_row):
    """
    Run the CoT zero-shot model on the input row using the provided model.

    :param model_name: The LLM model to use.
    :param input_row: The input row to classify
    :return: The entire response from the model.
    """
    system_text = ("You are a sentiment analysis model. You will analyze the text given by the user and provide two "
                   "outputs: 1) The sentiment of the text from five sentiment categories: Strongly Negative, "
                   "Slightly Negative, Neutral, Slightly Positive, and Strongly Positive. The output should be a "
                   "number from zero (0) to four (4), which represents the corresponding sentiment category. Follow "
                   "this rule: return only a number from zero four. 2) An explanation or reasoning for the results "
                   "in a separate paragraph. Do not combine the two outputs. The response should be structured as "
                   "follows: First, a separate explanation paragraph. Then, the number from zero to four representing "
                   "the sentiment category. Example response: The text contains mixed sentiments with a stronger "
                   "leaning towards neutrality. There are slight negative and positive sentiments detected with the "
                   "use of phrases such as \"inconvenient\", \"annoying\", and \"supporting\", but the overall tone "
                   "is neutral. 2")
    prompt = "Textual Input: " + input_row
    return OllamaCached.chain_of_reasoning_zero_shot(model_name, system_text, prompt)


def chain_of_thought_few(model_name, training_data, input_row):
    """
    Run the CoT few-shot model on the input row using the provided model.

    :param model_name: The LLM model to use.
    :param training_data: The training data to use for the CoT few-shot model.
    :param input_row: The input row to classify
    :return: The entire response from the model.
    """
    system_text = ("You are a sentiment analysis model. You will analyze the text given by the user and provide two "
                   "outputs: 1) The sentiment of the text from five sentiment categories: Strongly Negative, "
                   "Slightly Negative, Neutral, Slightly Positive, and Strongly Positive. The output should be a "
                   "number from zero (0) to four (4), which represents the corresponding sentiment category. Follow "
                   "this rule: return only a number from zero four. 2) An explanation or reasoning for the results "
                   "in a separate paragraph. Do not combine the two outputs. The response should be structured as "
                   "follows: First, a separate explanation paragraph. Then, the number from zero to four representing "
                   "the sentiment category. Example response: The text contains mixed sentiments with a stronger "
                   "leaning towards neutrality. There are slight negative and positive sentiments detected with the "
                   "use of phrases such as \"inconvenient\", \"annoying\", and \"supporting\", but the overall tone "
                   "is neutral. 2")
    prompt = "Textual Input: " + input_row

    # Format the training data and create explanations for it
    formatted_training = []
    for t in range(0, len(training_data)):
        training_row = training_data.iloc[t].values  # 0 = input, 1 = target hard label
        explanation = get_or_gen_explanation(model_name, training_row)
        explained_result = explanation + "\n " + str(training_row[1])
        formatted_training.append([training_row[0], explained_result])

    return OllamaCached.chain_of_reasoning_few_shot(model_name, system_text, formatted_training, prompt)


def get_or_gen_explanation(model_name, training_row):
    """
    Get the explanation from the CSV file or generate a new one.

    :param model_name: The model to use to generate the explanation
    :param training_row: The row to generate the explanation for
    :return: The explanation for the training row
    """
    # Open the explanations CSV file
    with open('../data/explanations_hard.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Go through the file and generate an explanation if one is missing, or return the existing one
    temp_exp = ""
    for row in rows:
        if row[0] == training_row[0]:
            try:
                return row[1]
            except IndexError:
                temp_exp = OllamaCached.generate_explanation_hard(model_name, training_row)
                row.append(temp_exp)
                break

    # Write the new explanation to the CSV file
    with open('../data/explanations_hard.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    # Return the new explanation
    return temp_exp


def write_to_file(name, data):
    """
    Write the data to a CSV file.

    :param name: The name of the file
    :param data: The data to write
    :return: Nothing
    """
    with open(name + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in data:
            csvwriter.writerow(row)


def extract_sentiment_value(result):
    """
    Extract the sentiment value from the result.

    :param result: The full result from the model
    :return: The sentiment value
    """
    integers = re.findall(r'\d+', result)
    return int(integers[-1]) if integers else -1


if __name__ == "__main__":
    """
    Run the models on the hard labels and write the results to CSV files.
    """
    # Set the model to use and generate the hard labels
    hard_labels = read_data_and_create_hard_labels()
    model = "llama3:latest"

    # Initialize the lists to store the results
    zero_shot_results = []
    few_shot_results = []
    chain_of_thought_zero_results = []
    chain_of_thought_few_results = []

    # Split into train/validate/test sets - 70% training, 30% testing
    train_df, test_df = train_test_split(hard_labels, test_size=0.3, random_state=42)

    # Run the models on the test data
    for i in range(0, len(test_df)):
        # Initialize the temporary variables
        input_and_target_results = test_df.iloc[i].values  # 0 = input, 1 = target hard label
        zero_shot_predicted = -1
        few_shot_predicted = -1
        chain_of_thought_zero_predicted = -1
        chain_of_thought_few_predicted = -1

        # Run the zero-shot model
        zero_result_extracted = -1
        while zero_result_extracted == -1:
            try:
                zero_result = zero_shot(model, input_and_target_results[0])
                zero_result_extracted = extract_sentiment_value(zero_result)
            except Exception as e:
                print(e)
                print("Oopsi, trying zero again!")
        zero_shot_predicted = str(zero_result_extracted)

        # Run the few-shot model
        few_result_extracted = -1
        while few_result_extracted == -1:
            try:
                few_result = few_shot(model, train_df, input_and_target_results[0])
                few_result_extracted = extract_sentiment_value(few_result)
            except Exception as e:
                print(e)
                print("Oopsi, trying few again!")
        few_shot_predicted = str(few_result_extracted)

        # Run the CoT zero-shot model
        cot_zero_result_extracted = -1
        while cot_zero_result_extracted == -1:
            try:
                cot_zero_result = chain_of_thought_zero(model, input_and_target_results[0])
                cot_zero_result_extracted = extract_sentiment_value(cot_zero_result)
            except Exception as e:
                print(e)
                print("Oopsi, trying cot_zero again!")
        chain_of_thought_zero_predicted = str(cot_zero_result_extracted)

        # Run the CoT few-shot model
        cot_few_result_extracted = -1
        while cot_few_result_extracted == -1:
            try:
                cot_few_result = chain_of_thought_few(model, train_df, input_and_target_results[0])
                cot_few_result_extracted = extract_sentiment_value(cot_few_result)
            except Exception as e:
                print(e)
                print("Oopsi, trying cot_few again!")
        chain_of_thought_few_predicted = str(cot_few_result_extracted)

        # Append the results to the lists
        zero_shot_results.append(zero_shot_predicted)
        few_shot_results.append(few_shot_predicted)
        chain_of_thought_zero_results.append(chain_of_thought_zero_predicted)
        chain_of_thought_few_results.append(chain_of_thought_few_predicted)

    # Write the results to CSV files
    write_to_file("zero_shot_res_hard", zero_shot_results)
    write_to_file("few_shot_res_hard", few_shot_results)
    write_to_file("cot_zero_res_hard", chain_of_thought_zero_results)
    write_to_file("cot_few_res_hard", chain_of_thought_few_results)
