from sklearn.model_selection import train_test_split
import pandas as pd
import OllamaCached
import csv


def read_data_and_create_subjectivity_labels():
    """
    Reads the data from the CSV files and creates subjectivity labels.

    :return: A dataframe with the textual inputs and the subjectivity labels.
    """
    # Read data from all the CSV files with the labels of all the annotators
    data_timur = pd.read_csv("../data/timur.csv")
    data_adina = pd.read_csv("../data/adina.csv")
    data_bente = pd.read_csv("../data/bente.csv")
    data_ana = pd.read_csv("../data/ana.csv")
    data_joosje = pd.read_csv("../data/joosje.csv")

    # Extract labels from reach annotated row
    text_input = data_timur.iloc[:, 5]
    col6_timur = data_timur.iloc[:, 6]
    col6_adina = data_adina.iloc[:, 6]
    col6_bente = data_bente.iloc[:, 6]
    col6_ana = data_ana.iloc[:, 6]
    col6_joosje = data_joosje.iloc[:, 6]

    # Combine to create one dataframe and return it
    return pd.concat([
        text_input, col6_timur, col6_adina, col6_bente, col6_ana, col6_joosje
    ], axis=1)


def few_shot(model_name, training_data, input_row):
    """
    Run the few-shot model on the input row using the provided model.

    :param model_name: The LLM model to use.
    :param training_data: The training data to use for the few-shot model.
    :param input_row: The input row to classify
    :return: The entire response from the model.
    """
    system_text = ("You are a sentiment analysis model. Analyze the user's text and provide the following output: The "
                   "sentiment of the text categorized into one of five categories: Strongly Negative, Slightly "
                   "Negative, Neutral, Slightly Positive, and Strongly Positive. Provide an array of five numbers, "
                   "each representing the predicted sentiment category by five different annotators. Each number "
                   "should be between zero (0) and four (4), corresponding to the sentiment categories. Consider the "
                   "provided history to predict each annotator's sentiment annotation for the new text. Format the "
                   "output as a Python array: [annotator 1, annotator 2, ..., annotator 5]. Example response: "
                   "[3, 2, 4, 1, 0]")

    prompt = "Textual Input: " + input_row

    # Format the training data
    formatted_training = []
    for t in range(0, len(training_data)):
        training_row = training_data.iloc[t].values  # 0 = input, 5 next values are target soft labels
        correct_result = "[" + ", ".join(str(x) for x in training_row[1:6]) + "]"
        formatted_training.append([training_row[0], correct_result])

    return OllamaCached.few_shot(model_name, system_text, formatted_training, prompt)


def chain_of_thought_few(model_name, training_data, input_row):
    """
    Run the CoT few-shot model on the input row using the provided model.

    :param model_name: The LLM model to use.
    :param training_data: The training data to use for the CoT few-shot model.
    :param input_row: The input row to classify
    :return: The entire response from the model.
    """
    system_text = ("You are a sentiment analysis model. Analyze the user's text and provide two outputs: 1) The "
                   "sentiment of the text categorized into one of five categories: Strongly Negative, Slightly "
                   "Negative, Neutral, Slightly Positive, and Strongly Positive. Provide an array of five numbers, "
                   "each representing the predicted sentiment category by five different annotators. Each number "
                   "should be between zero (0) and four (4), corresponding to the sentiment categories. Consider the "
                   "provided history to predict each annotator's sentiment annotation for the new text. Format the "
                   "output as a Python array: [annotator 1, annotator 2, ..., annotator 5]. 2) An explanation or "
                   "reasoning for the results in a separate paragraph. Do not combine the two outputs. The response "
                   "should be structured as follows: First, the explanation paragraph. Then, the array with five "
                   "numbers representing the predictions of the sentiment category for each annotator. Example "
                   "response: The text contains mixed sentiments with a stronger leaning towards neutrality. There are "
                   "slight negative and positive sentiments detected with the use of phrases such as 'inconvenient', "
                   "'annoying', and 'supporting', but the overall tone is neutral. Annotator 1 generally leans towards "
                   "a slightly negative skew, whereas Annotator 2 skews positive. The other annotators all skew "
                   "neither way, but 3 and 5 always have the same results. [1, 3, 2, 3, 2]")

    prompt = "Textual Input: " + input_row

    # Format the training data and get or generate explanations
    formatted_training = []
    for t in range(0, len(training_data)):
        training_row = training_data.iloc[t].values  # 0 = input, 5 next values are target soft labels
        explanation = get_or_gen_explanation(model_name, training_row, training_data)
        correct_result = "[" + ", ".join(str(x) for x in training_row[1:6]) + "]"
        explained_result = explanation + "\n " + correct_result
        formatted_training.append([training_row[0], explained_result])

    return OllamaCached.chain_of_reasoning_few_shot(model_name, system_text, formatted_training, prompt)


def get_or_gen_explanation(model_name, training_row, training_data):
    """
    Get the explanation from the CSV file or generate a new one.

    :param model_name: The model to use to generate the explanation
    :param training_row: The row to generate the explanation for
    :param training_data: All the training data available
    :return: The explanation for the training row
    """
    # Open the explanations CSV file
    with open('../data/explanations_subjectivity.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Go through the file and generate an explanation if one is missing, or return the existing one
    temp_exp = ""
    for row in rows:
        if row[0] == training_row[0]:
            try:
                return row[1]
            except IndexError:
                temp_exp = OllamaCached.generate_explanation_subjectivity(model_name, training_row, training_data)
                row.append(temp_exp)
                break

    # Write the new explanation to the CSV file
    with open('../data/explanations_subjectivity.csv', 'w', newline='') as file:
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
    Run the models on subjectivity labels and write the results to CSV files.
    """
    # Set the model to use and generate the subjectivity labels
    subjectivity_labels = read_data_and_create_subjectivity_labels()
    model = "llama3:latest"

    # Initialize the lists to store the results
    few_shot_results = []
    chain_of_thought_few_results = []

    # Split into train/validate/test sets - 70% training, 30% testing
    train_df, test_df = train_test_split(subjectivity_labels, test_size=0.3, random_state=42)

    # Run the models on the test data
    for i in range(0, len(test_df)):
        # Initialize the temporary variables
        input_and_target_results = test_df.iloc[i].values  # 0 = input, 1-5 = annotator's labels

        # Run the Few-Shot model
        few_shot_result = few_shot(model, train_df, input_and_target_results[0])
        few_shot_result_extracted = extract_python_array(few_shot_result)
        few_shot_results.append(few_shot_result_extracted)

        # Run the CoT Few-Shot model
        chain_of_thought_few_result = chain_of_thought_few(model, train_df, input_and_target_results[0])
        chain_of_thought_few_result_extracted = extract_python_array(chain_of_thought_few_result)
        chain_of_thought_few_results.append(chain_of_thought_few_result_extracted)

    # Write the results to CSV files
    write_to_file("few_shot_res_subjectivity", few_shot_results)
    write_to_file("cot_few_res_subjectivity", chain_of_thought_few_results)
