import array
import os
from tqdm import tqdm
import ollama

# Set environment variable to save models to the models folder in the repository.
os.environ["OLLAMA_MODELS"] = "./models"
os.environ["OLLAMA_KEEP_ALIVE"] = "1m"


def check_and_download_model(model_name: str) -> None:
    """
    Checks if the specified LLM exists locally and downloads it if necessary.

    Args:
        model_name (str): Tag of the LLM.
    """
    model_exists: bool = False

    existing_models = ollama.list()["models"]
    for model in existing_models:
        if model["name"] == model_name:
            model_exists = True

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
    Chat with the llm without any prior training (i.e. zero-shot)

    Args:
        model_name (str): Tag of the LLM.
        sys_message (str): The prompt to provide to the LLM as the system.
        message (str): The prompt to provide to the LLM as the user.
    Returns:
        response (str): The response from the assistant.
    """
    check_and_download_model(model_name)

    print("Zero Shot (" + model_name + "):")

    initial_messages = [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": message}
    ]
    response = ollama.chat(model=model_name,
                           messages=initial_messages,
                           stream=False)
    return response["message"]["content"]


def few_shot(model_name: str, sys_message: str, training_messages: array, message: str) -> str:
    """
    Chat with the llm and provide some training samples for few-shot learning.

    Args:
        model_name (str): Tag of the LLM.
        sys_message (str): The prompt to provide to the LLM as the system.
        training_messages (array): The training messages provided as (input, output) pairs.
        message (str): The prompt to provide to the LLM as the user.
    Returns:
        response (str): The response from the assistant.
    """
    check_and_download_model(model_name)

    print("Few Shot (" + model_name + "): " + str(len(training_messages)) + " Samples")
    initial_messages = [
        {"role": "system", "content": sys_message}
    ]
    for training_message in training_messages:
        user_temp = {"role": "user", "content": training_message[0]}
        assistant_temp = {"role": "assistant", "content": training_message[1]}
        initial_messages = initial_messages.append(user_temp)
        initial_messages = initial_messages.append(assistant_temp)

    initial_messages = initial_messages.append({"role": "user", "content": message})

    response = ollama.chat(model=model_name,
                           messages=initial_messages,
                           stream=False)
    return response["message"]["content"]


def chain_of_reasoning_zero_shot(model_name: str, sys_message: str, message: str) -> str:
    """
    Chat with the llm without any prior training (i.e. zero-shot), but with chain of thought reasoning.

    Args:
        model_name (str): Tag of the LLM.
        sys_message (str): The prompt to provide to the LLM as the system.
        message (str): The prompt to provide to the LLM as the user.
    Returns:
        response (str): The response from the assistant.
    """
    check_and_download_model(model_name)

    print("Chain of Reasoning - Zero Shot (" + model_name + "):")

    initial_messages = [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": message},
        {"role": "assistant", "content": "Let's think step by step."}
    ]
    complex_response = ollama.chat(model=model_name,
                                   messages=initial_messages,
                                   stream=False)
    response = ollama.chat(model=model_name,
                           messages=[
                               initial_messages,
                               complex_response["message"],
                               {"role": "assistant", "content": "Therefore, the final answer is "}
                           ],
                           stream=False)

    return response["message"]["content"]


def chain_of_reasoning_few_shot(model_name: str, sys_message: str, training_messages: array, message: str) -> str:
    """
    Chat with the llm and provide some training samples for few-shot learning, but also with chain of thought reasoning.

    Args:
        model_name (str): Tag of the LLM.
        sys_message (str): The prompt to provide to the LLM as the system.
        training_messages (array): The training messages provided as (input, output) pairs.
        message (str): The prompt to provide to the LLM as the user.
    Returns:
        response (str): The response from the assistant.
    """
    check_and_download_model(model_name)

    print("Few Shot (" + model_name + "): " + str(len(training_messages)) + " Samples")
    initial_messages = [
        {"role": "system", "content": sys_message}
    ]
    for training_message in training_messages:
        user_temp = {"role": "user", "content": training_message[0]}
        assistant_temp = {"role": "assistant", "content": training_message[1]}
        initial_messages = initial_messages.append(user_temp)
        initial_messages = initial_messages.append(assistant_temp)

    initial_messages = initial_messages.append({"role": "user", "content": message})
    initial_messages = initial_messages.append({"role": "assistant", "content": "Let's think step by step."})

    complex_response = ollama.chat(model=model_name,
                                   messages=initial_messages,
                                   stream=False)
    response = ollama.chat(model=model_name,
                           messages=[
                               initial_messages,
                               complex_response["message"],
                               {"role": "assistant", "content": "Therefore, the final answer is "}
                           ],
                           stream=False)

    return response["message"]["content"]
