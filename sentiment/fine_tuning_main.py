from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


if __name__ == "__main__":
    """
    Run this script to fine-tune the Meta-Llama model on the sentiment analysis dataset, to generate a fine-tuned model.
    """
    # Loading Dataset
    dataset = load_dataset('../data/combined_data.csv')

    # Loading Model
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

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

    # Prepare Training
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

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # Start Training
    trainer.train()

    # Save the Model
    model.save_pretrained("../models/llama-3-sentiment-finetuned")
    tokenizer.save_pretrained("../models/llama-3-sentiment-finetuned")
