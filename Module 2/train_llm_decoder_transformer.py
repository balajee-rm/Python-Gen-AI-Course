import torch
from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from transformer_decoder import SimpleScratchDecoder

# 1. Load Khan Academy educational content and GPT-2 tokenizer
print("Loading Khan Academy dataset...")
raw_dataset = load_dataset("HuggingFaceTB/cosmopedia", "khanacademy", split="train")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Create the from-scratch decoder
model = SimpleScratchDecoder(vocab_size=len(tokenizer), d_model=128, num_heads=4)


# 3. Tokenize educational content
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")


# Use 24,000 samples for training
MODEL_DIR = "./llm_decoder_transformer_model"
small_dataset = raw_dataset.select(range(24000))
tokenized_dataset = small_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# 4. Train the model
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
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
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# Save the decoder weights and tokenizer in the same directory
torch.save(model.state_dict(), f"{MODEL_DIR}/model_state.pt")
tokenizer.save_pretrained(MODEL_DIR)
