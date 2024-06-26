from sklearn.model_selection import train_test_split
import pandas as pd
import OllamaCached
import operator
import csv


def read_data_and_create_soft_labels():
    """
    Reads the data from the csv files and creates soft labels for the data.

    :return: A dataframe with the soft labels for the data.
    """
    # Read the data from the csv files for each annotator
    data_timur = pd.read_csv("../data/timur.csv")
    data_adina = pd.read_csv("../data/adina.csv")
    data_bente = pd.read_csv("../data/bente.csv")
    data_ana = pd.read_csv("../data/ana.csv")
    data_joosje = pd.read_csv("../data/joosje.csv")
    all_data = [data_timur, data_adina, data_bente, data_ana, data_joosje]

    # Initialize the array for the soft labels
    soft_labelled_data = []

    # Generate the soft labels for each data point
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

    # Return the soft labelled data
    return pd.DataFrame(
        soft_labelled_data,
        None,
        ["Datapoint", "Strongly Negative", "Slightly Negative", "Neutral", "Slightly Positive", "Strongly Positive"]
    )


def zero_shot(model_name, input_row):
    """
    Run the zero-shot model on the input row.

    :param model_name: The name of the LLM model to use.
    :param input_row: The input row to run the model on.
    :return: The response from the LLM
    """
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
    """
    Run the few-shot model on the input row.

    :param model_name: The name of the LLM model to use.
    :param training_data: The training data to use for the few-shot model.
    :param input_row: The input row to run the model on.
    :return: The response from the LLM
    """
    system_text = ("You are a sentiment analysis model. You will analyze the text given by the user and provide a "
                   "probability distribution across five sentiment categories: Strongly Negative, Slightly Negative, "
                   "Neutral, Slightly Positive, and Strongly Positive. The output should be a Python list of floats "
                   "(e.g., [0.2, 0.3, 0.4, 0.05, 0.05]), where each element represents the probability of the "
                   "corresponding sentiment category. The sum of all probabilities must equal 1.0. Follow these "
                   "rules: 1) Return only the Python list of floats. 2) Ensure the sum of the probabilities equals "
                   "1.0. If not, adjust the values proportionally. 3) Do not provide any additional information or "
                   "text in your response. Example response: [0.2, 0.3, 0.4, 0.05, 0.05]")
    prompt = "Textual Input: " + input_row

    # Format the training data
    formatted_training = []
    for t in range(0, len(training_data)):
        training_row = training_data.iloc[t].values  # 0 = input, 5 next values are target soft labels
        correct_result = "[" + ", ".join(str(x) for x in training_row[1:6]) + "]"
        formatted_training.append([training_row[0], correct_result])

    return OllamaCached.few_shot(model_name, system_text, formatted_training, prompt)


def chain_of_thought_zero(model_name, input_row):
    """
    Run the CoT zero-shot model on the input row.

    :param model_name: The name of the LLM model to use.
    :param input_row: The input row to run the model on.
    :return: The response from the LLM
    """
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
    """
    Run the CoT few-shot model on the input row.

    :param model_name: The name of the LLM model to use.
    :param training_data: The training data to use for the few-shot model.
    :param input_row: The input row to run the model on.
    :return: The response from the LLM
    """
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

    # Format the training data and get or generate explanations
    formatted_training = []
    for t in range(0, len(training_data)):
        training_row = training_data.iloc[t].values  # 0 = input, 5 next values are target soft labels
        explanation = get_or_gen_explanation(model_name, training_row)
        correct_result = "[" + ", ".join(str(x) for x in training_row[1:6]) + "]"
        explained_result = explanation + "\n " + correct_result
        formatted_training.append([training_row[0], explained_result])

    return OllamaCached.chain_of_reasoning_few_shot(model_name, system_text, formatted_training, prompt)


def get_or_gen_explanation(model_name, training_row):
    """
    Get or generate an explanation for the training row.

    :param model_name: The name of the LLM model to use to generate the explanation.
    :param training_row: The training row to get or generate the explanation for.
    :return: The explanation for the training row
    """
    # Read the explanations from the csv file
    with open('../data/explanations_soft.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Check if the explanation is already generated
    temp_exp = ""
    for row in rows:
        if row[0] == training_row[0]:
            try:
                return row[1]
            except IndexError:
                temp_exp = OllamaCached.generate_explanation_soft(model_name, training_row)
                row.append(temp_exp)
                break

    # Generate the explanation and write it to the csv file
    with open('../data/explanations_soft.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    # Return the generated explanation
    return temp_exp


def write_to_file(name, data):
    """
    Write the data to a csv file.

    :param name: The filename to write the data to.
    :param data: The data to write to the file.
    :return: Nothing.
    """
    with open(name + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in data:
            csvwriter.writerow(row)


def extract_python_array(result):
    """
    Extract the python array from the LLM response / result.

    :param result: The result to extract the python array from.
    :return: The python array extracted from the result.
    """
    string_arr = result.split("[")[1].split("]")[0].split(", ")

    # Convert the string array to a float array
    for s in range(0, len(string_arr)):
        string_arr[s] = float(string_arr[s])

    # Return the float array
    return string_arr


if __name__ == "__main__":
    """
    Run the models on the soft labels and write the results to CSV files.
    """
    # Set the model to use and generate the soft labels
    soft_labels = read_data_and_create_soft_labels()
    model = "llama3:latest"

    # Initialize the results arrays
    zero_shot_results = []
    few_shot_results = []
    chain_of_thought_zero_results = []
    chain_of_thought_few_results = []

    # Split into train/validate/test sets - 70% training, 30% testing
    train_df, test_df = train_test_split(soft_labels, test_size=0.3, random_state=42)

    # Run the models on the test data
    for i in range(0, len(test_df)):
        # Initialize the temporary results arrays
        input_and_target_results = test_df.iloc[i].values  # 0 = input, 5 next values are target soft labels

        # Run the zero-shot model
        zero_result_extracted = ""
        while zero_result_extracted == "":
            try:
                zero_result = zero_shot(model, input_and_target_results[0])
                zero_result_extracted = extract_python_array(zero_result)
            except Exception as e:
                print(e)
                print("Oopsi, trying zero again!")
        zero_shot_predicted = zero_result_extracted

        # Run the few-shot model
        few_result_extracted = ""
        while few_result_extracted == "":
            try:
                few_result = few_shot(model, train_df, input_and_target_results[0])
                few_result_extracted = extract_python_array(few_result)
            except Exception as e:
                print(e)
                print("Oopsi, trying few again!")
        few_shot_predicted = few_result_extracted

        # Run the CoT zero-shot model
        cot_zero_result_extracted = ""
        while cot_zero_result_extracted == "":
            try:
                cot_zero_result = chain_of_thought_zero(model, input_and_target_results[0])
                cot_zero_result_extracted = extract_python_array(cot_zero_result)
            except Exception as e:
                print(e)
                print("Oopsi, trying cot_zero again!")
        chain_of_thought_zero_predicted = cot_zero_result_extracted

        # Run the CoT few-shot model
        cot_few_result_extracted = ""
        while cot_few_result_extracted == "":
            try:
                cot_few_result = chain_of_thought_few(model, train_df, input_and_target_results[0])
                cot_few_result_extracted = extract_python_array(cot_few_result)
            except Exception as e:
                print(e)
                print("Oopsi, trying cot_few again!")
        chain_of_thought_few_predicted = cot_few_result_extracted

        # Append the results to the results arrays
        zero_shot_results.append(zero_shot_predicted)
        few_shot_results.append(few_shot_predicted)
        chain_of_thought_zero_results.append(chain_of_thought_zero_predicted)
        chain_of_thought_few_results.append(chain_of_thought_few_predicted)

    # Write the results to csv files
    write_to_file("zero_shot_res_soft", zero_shot_results)
    write_to_file("few_shot_res_soft", few_shot_results)
    write_to_file("cot_zero_res_soft", chain_of_thought_zero_results)
    write_to_file("cot_few_res_soft", chain_of_thought_few_results)
