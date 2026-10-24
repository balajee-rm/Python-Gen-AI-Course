import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# 1. Tokenizer & Blank Model Setup
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=256,
    n_embd=128,
    n_layer=4,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)

# 2. Load TinyStories (Clean, simple dataset for small models)
print("Loading TinyStories dataset...")
raw_dataset = load_dataset("roneneldan/TinyStories", split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

# Use 10,000 samples for better learning
small_dataset = raw_dataset.select(range(10000))
tokenized_dataset = small_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

# 3. Training
training_args = TrainingArguments(
    output_dir="./my_laptop_llm_tinystories",
    num_train_epochs=5,               # More passes over data
    per_device_train_batch_size=8,
    save_steps=500,
    logging_steps=50,
    learning_rate=5e-4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained("./my_laptop_llm_tinystories")
tokenizer.save_pretrained("./my_laptop_llm_tinystories")