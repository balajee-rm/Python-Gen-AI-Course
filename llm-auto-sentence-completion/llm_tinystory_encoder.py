from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# 1. Tokenizer & Blank Model Setup
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

config = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=128,
    pad_token_id=tokenizer.pad_token_id,
)
model = BertForMaskedLM(config)

# 2. Load TinyStories (Clean, simple dataset for small models)
print("Loading TinyStories dataset...")
raw_dataset = load_dataset("roneneldan/TinyStories", split="train")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")


# Use 10,000 samples for better learning
small_dataset = raw_dataset.select(range(10000))
tokenized_dataset = small_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Training
training_args = TrainingArguments(
    output_dir="./my_laptop_llm_tinystories_encoder",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    logging_steps=50,
    learning_rate=5e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
)

trainer.train()

model.save_pretrained("./my_laptop_llm_tinystories_encoder")
tokenizer.save_pretrained("./my_laptop_llm_tinystories_encoder")
