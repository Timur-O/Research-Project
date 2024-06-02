from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from torch import tensor
import pandas as pd


class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['input']
        label = row['label']
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
        inputs['labels'] = tensor([label]).long()
        return inputs


if __name__ == "__main__":
    """
    Run this script to fine-tune the Meta-Llama model on the sentiment analysis dataset, to generate a fine-tuned model.
    """
    # Loading Dataset
    dataframe = pd.read_csv('../data/combined_data_hard.csv')

    # Split the dataset into training, validation, and test sets
    train_df, temp_df = train_test_split(dataframe, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Load the Meta-Llama model and tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load your prepared data
    train_dataset = SentimentDataset(train_df, tokenizer)
    val_dataset = SentimentDataset(val_df, tokenizer)
    test_dataset = SentimentDataset(test_df, tokenizer)

    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,  # Scaling parameter
        target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA
        lora_dropout=0.1,  # Dropout probability
        bias="none"  # Bias configuration
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Prepare the training
    training_args = TrainingArguments(
        output_dir="../models/llama-3-sentiment-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=10_000,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start the training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("../models/llama-3-sentiment-finetuned")
    tokenizer.save_pretrained("../models/llama-3-sentiment-finetuned")
