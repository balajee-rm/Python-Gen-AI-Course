import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# ---------------------------------------------------------
# 1. Initialize Tokenizer & Model Architecture
# ---------------------------------------------------------
print("[1/5] Setting up tokenizer and model architecture...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=256,
    n_ctx=256,
    n_embd=128,
    n_layer=4,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)

# ---------------------------------------------------------
# 2. Load & Tokenize Real Dataset (WikiText-2)
# ---------------------------------------------------------
print("[2/5] Loading real text dataset (WikiText-2)...")
# Load train split of WikiText-2 using full namespace
raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

# Filter out empty lines
raw_dataset = raw_dataset.filter(lambda example: len(example["text"].strip()) > 0)

# Tokenize text
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
        padding="max_length"
    )

# Select a small slice (2,000 samples) for fast laptop training
small_dataset = raw_dataset.select(range(2000))
tokenized_dataset = small_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Format dataset for PyTorch Trainer
tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

# ---------------------------------------------------------
# 3. Configure Trainer & Arguments
# ---------------------------------------------------------
print("[3/5] Configuring training options...")
training_args = TrainingArguments(
    output_dir="./my_laptop_llm",
    num_train_epochs=3,               # 3 epochs over 2,000 samples
    per_device_train_batch_size=8,     # Increased batch size slightly
    save_steps=200,
    logging_steps=20,
    learning_rate=5e-4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# ---------------------------------------------------------
# 4. Train Model & Plot Loss
# ---------------------------------------------------------
print("[4/5] Starting model training on real data...")
trainer.train()

# Plot Loss Graph
log_history = trainer.state.log_history
steps = [entry["step"] for entry in log_history if "loss" in entry]
losses = [entry["loss"] for entry in log_history if "loss" in entry]

plt.figure(figsize=(8, 4))
plt.plot(steps, losses, label="Training Loss", color="green", marker="o")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("WikiText Training Loss vs. Steps")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. Save Model
# ---------------------------------------------------------
print("[5/5] Saving model...")
model.save_pretrained("./my_laptop_llm2")
tokenizer.save_pretrained("./my_laptop_llm2")
print("Model saved successfully to ./my_laptop_llm2")