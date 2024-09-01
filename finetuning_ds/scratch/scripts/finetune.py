import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainerCallback
import torch
import transformers




def main():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-step-50K-105b')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Load dataset from the Hugging Face datasets library
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


    # Tokenize the texts
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


    # Apply "tokenize_function" to all the examples in iterator object "dataset". 
    # "batched=True" implies passing examples in a batch instead of one at a time
    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    # Load the data collator (the batch maker given a dataset)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # TinyLlama uses a causal (not masked) language model, similar to GPT-2
    )


    # Load the model
    model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-step-50K-105b')
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=1000,
        save_total_limit=2,
        fp16=True, 
        deepspeed=os.path.join(os.environ['PWD'],os.environ['DS_CONFIG']),  # Path to DeepSpeed config file        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':True},
        report_to='tensorboard',
        logging_dir='./logs',
        logging_steps=10,
        logging_strategy='steps',
        log_level='info',
    )


    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"]
    )


    # Start the training
    trainer.train()


    # Save the final model and tokenizer
    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_model')


if __name__ == "__main__":
    main()
