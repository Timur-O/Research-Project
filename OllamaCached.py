from tqdm import tqdm
import ollama
import array


def check_and_download_model(model_name: str) -> None:
    """
    Checks if the specified LLM exists locally and downloads it if necessary.

    Args:
        model_name (str): Tag of the LLM.
    """
    model_exists: bool = False

    # Check the existing models
    existing_models = ollama.list()["models"]
    for model in existing_models:
        if model["name"] == model_name:
            model_exists = True

    # Download the model if it does not exist
    if not model_exists:
        print("Model Uncached - Downloading...")
        current_digest, bars = '', {}
        for progress in ollama.pull(model_name, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get('status'))
                continue

            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest
        print("Download Complete.")
    else:
        print("Model Cached - Using Cached Version...")


def zero_shot(model_name: str, sys_message: str, message: str) -> str:
    """
    Send a message to the llm without any prior training (i.e. zero-shot)

    Args:
        model_name (str): Tag of the LLM.
        sys_message (str): The prompt to provide to the LLM as the system.
        message (str): The prompt to provide to the LLM as the user.
    Returns:
        response (str): The response from the assistant.
    """
    print("Zero Shot...")

    check_and_download_model(model_name)
    initial_messages = [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": message}
    ]

    # Send the message and get the response from the assistant
    response = ollama.chat(model=model_name,
                           messages=initial_messages,
                           stream=False)
    return response["message"]["content"]


def few_shot(model_name: str, sys_message: str, training_messages: array, message: str) -> str:
    """
    Send a message to the llm and provide some training samples for few-shot learning.

    Args:
        model_name (str): Tag of the LLM.
        sys_message (str): The prompt to provide to the LLM as the system.
        training_messages (array): The training messages provided as (input, output) pairs.
        message (str): The prompt to provide to the LLM as the user.
    Returns:
        response (str): The response from the assistant.
    """
    print("Few Shot...")

    # Prepare the training history
    check_and_download_model(model_name)
    initial_messages = [
        {"role": "system", "content": sys_message}
    ]
    for training_message in training_messages:
        user_temp = {"role": "user", "content": training_message[0]}
        assistant_temp = {"role": "assistant", "content": str(training_message[1])}
        initial_messages.append(user_temp)
        initial_messages.append(assistant_temp)
    initial_messages.append({"role": "user", "content": message})

    # Send the message and get the response from the assistant
    response = ollama.chat(model=model_name,
                           messages=initial_messages,
                           stream=False)
    return response["message"]["content"]


def chain_of_reasoning_zero_shot(model_name: str, sys_message: str, message: str) -> str:
    """
    Send a message to the llm without any prior training (i.e. zero-shot), but with chain of thought reasoning.

    Args:
        model_name (str): Tag of the LLM.
        sys_message (str): The prompt to provide to the LLM as the system.
        message (str): The prompt to provide to the LLM as the user.
    Returns:
        response (str): The response from the assistant.
    """
    print("Chain-of-Thought - Zero-Shot...")

    check_and_download_model(model_name)
    initial_messages = [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": message}
    ]

    # Send the message and get the response from the assistant
    complex_response = ollama.chat(model=model_name,
                                   messages=initial_messages,
                                   stream=False)
    return complex_response["message"]["content"]


def chain_of_reasoning_few_shot(model_name: str, sys_message: str, training_messages: array, message: str) -> str:
    """
    Send a message to the llm and provide some training samples for few-shot learning, but also with chain of thought
    reasoning.

    Args:
        model_name (str): Tag of the LLM.
        sys_message (str): The prompt to provide to the LLM as the system.
        training_messages (array): The training messages provided as (input, output) pairs.
        message (str): The prompt to provide to the LLM as the user.
    Returns:
        response (str): The response from the assistant.
    """
    print("Chain-Of-Thought Few-Shot...")

    # Prepare the training history
    check_and_download_model(model_name)
    initial_messages = [
        {"role": "system", "content": sys_message}
    ]
    for training_message in training_messages:
        user_temp = {"role": "user", "content": training_message[0]}
        assistant_temp = {"role": "assistant", "content": training_message[1]}
        initial_messages.append(user_temp)
        initial_messages.append(assistant_temp)
    initial_messages.append({"role": "user", "content": message})

    # Send the message and get the response from the assistant
    complex_response = ollama.chat(model=model_name,
                                   messages=initial_messages,
                                   stream=False)
    return complex_response["message"]["content"]


def generate_explanation_soft(model_name, to_explain):
    """
    Generate an explanation given a sentiment and a result in a soft-label setting.

    Args:
        model_name: The name of the LLM model
        to_explain: The input to explain [0] and the correct labels [1-5]
    Returns:
        The explanation of how this result is achieved.
    """
    system_prompt = ("You are an explanation generation model. Given an input text and its corresponding sentiment "
                     "probability distribution, your task is to craft a concise, three-sentence explanation that "
                     "clarifies why the input text aligns with the predicted sentiment. The probability distribution "
                     "follows this structure: Value 1: Strongly Negative, Value 2: Slightly Negative, Value 3: Neutral,"
                     "Value 4: Slightly Positive, Value 5: Strongly Positive. Your explanation should be informative, "
                     "focusing on the key words or phrases in the input text that most strongly contribute to the "
                     "predicted sentiment. Answer with only the explanation.")
    initial_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Input: " + to_explain[0] + " / Result to Explain: " + str(to_explain[1:5])}
    ]

    # Send the message and get the response from the assistant
    response = ollama.chat(model=model_name,
                           messages=initial_messages,
                           stream=False)
    return response["message"]["content"]


def generate_explanation_hard(model_name, to_explain):
    """
    Generate an explanation given a sentiment and a result for a hard-label setting.

    Args:
        model_name: The name of the LLM model
        to_explain: The input to explain [0] and the correct label [1]
    Returns:
        The explanation of how this result is achieved.
    """
    system_prompt = ("You are an explanation generation model. Given an input text and its corresponding sentiment, "
                     "your task is to craft a concise, three-sentence explanation that clarifies why the input text "
                     "aligns with the predicted sentiment. The sentiment label corresponds to the following: 0: "
                     "Strongly Negative, 1: Slightly Negative, 2: Neutral, 3: Slightly Positive, 4: Strongly Positive. "
                     "Your explanation should be informative, focusing on the key words or phrases in the input text "
                     "that most strongly contribute to the predicted sentiment. Answer with only the explanation.")
    initial_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Input: " + to_explain[0] + " / Result to Explain: " + str(to_explain[1])}
    ]

    # Send the message and get the response from the assistant
    response = ollama.chat(model=model_name,
                           messages=initial_messages,
                           stream=False)
    return response["message"]["content"]


def generate_explanation_subjectivity(model_name, to_explain, training_data):
    """
    Generate an explanation given a sentiment and a result for a subjectivity setting.

    Args:
        model_name: The name of the LLM model
        to_explain: The input to explain [0] and the correct label [1]
        training_data: The training data to provide to the LLM
    Returns:
        The explanation of how this result is achieved.
    """
    system_prompt = (
        "You are an explanation generation model. Given an input text and its corresponding sentiment annotations from "
        "five different annotators, your task is to craft a concise, three-sentence explanation clarifying why the "
        "input text aligns with the predicted sentiment annotations. The sentiment labels are: 0: Strongly Negative, "
        "1: Slightly Negative, 2: Neutral, 3: Slightly Positive, 4: Strongly Positive. The array is organized as "
        "follows: [Annotator 1, Annotator 2, ..., Annotator 5]. Your explanation should be informative and focus on "
        "the key words or phrases in the input text, as well as the history of predictions by each annotator, which "
        "are provided below, that most strongly contribute to the predicted sentiment. Provide only the explanation."
    )

    initial_messages = [
        {"role": "system", "content": system_prompt},
    ]

    for t in range(0, len(training_data)):
        temp_row = training_data.iloc[t].values  # 0 = input, 5 next values are target soft labels
        corr_result = "[" + ", ".join(str(x) for x in temp_row[1:6]) + "]"
        full_row = temp_row[0] + "\n " + corr_result
        initial_messages.append({"role": "system", "content": full_row})

    to_explain_res = "[" + ", ".join(str(x) for x in to_explain[1:6]) + "]"
    initial_messages.append({
        "role": "user", "content": "Input: " + to_explain[0]+ " / Result to Explain: " + str(to_explain_res)
    })

    # Send the message and get the response from the assistant
    response = ollama.chat(model=model_name,
                           messages=initial_messages,
                           stream=False)
    return response["message"]["content"]
