import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)

# ---------------------------------------------------------
# 1. Initialize Tokenizer & Model Architecture
# ---------------------------------------------------------
print("[1/5] Setting up tokenizer and model architecture...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Define a lightweight 2.5 Million parameter Transformer architecture
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
# 2. Prepare Small Custom Training Dataset
# ---------------------------------------------------------
print("[2/5] Preparing training dataset...")
raw_text = [
    "Artificial intelligence is transforming technology.",
    "Python is a popular programming language for machine learning.",
    "Vision language models combine computer vision and text processing.",
    "Training a small model on a laptop is a great learning exercise."
] * 50  # 200 total samples

# Tokenize texts into PyTorch tensors
inputs = tokenizer(
    raw_text,
    padding="max_length",
    max_length=64,
    truncation=True,
    return_tensors="pt"
)

# Create Hugging Face Dataset
dataset = Dataset.from_dict({
    "input_ids": inputs["input_ids"],
    "labels": inputs["input_ids"]
})

# ---------------------------------------------------------
# 3. Configure Trainer & Arguments
# ---------------------------------------------------------
print("[3/5] Configuring training options...")
training_args = TrainingArguments(
    output_dir="./my_laptop_llm",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=100,
    logging_steps=10,             # Log loss every 10 steps
    learning_rate=5e-4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# ---------------------------------------------------------
# 4. Train the Model & Plot Loss Curve
# ---------------------------------------------------------
print("[4/5] Starting model training...")
trainer.train()

# Plot Loss Graph using Matplotlib
log_history = trainer.state.log_history
steps = [entry["step"] for entry in log_history if "loss" in entry]
losses = [entry["loss"] for entry in log_history if "loss" in entry]

plt.figure(figsize=(8, 4))
plt.plot(steps, losses, label="Training Loss", color="blue", marker="o")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss vs. Steps (Learning Curve)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. Save Model and Run Inference
# ---------------------------------------------------------
print("[5/5] Saving model and running text generation test...")
model.save_pretrained("./my_laptop_llm")
tokenizer.save_pretrained("./my_laptop_llm")
print("Model saved successfully to ./my_laptop_llm")

# Test Text Generation (Inference)
generator = pipeline(
    "text-generation", 
    model="./my_laptop_llm", 
    tokenizer="./my_laptop_llm"
)

prompt = "Python is"
output = generator(
    prompt, 
    max_new_tokens=20,  # Limits new generated tokens so total index < 256
    num_return_sequences=1, 
    pad_token_id=tokenizer.eos_token_id
)

print("\n--- GENERATION TEST ---")
print("Prompt:", prompt)
print("Output:", output[0]["generated_text"])