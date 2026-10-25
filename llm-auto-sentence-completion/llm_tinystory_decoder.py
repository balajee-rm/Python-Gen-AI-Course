import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from transformer_decoder import SimpleScratchDecoder

# 1. Load TinyStories and GPT-2 tokenizer
print("Loading TinyStories dataset...")
raw_dataset = load_dataset("roneneldan/TinyStories", split="train")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# MODEL_DIR = "./my_laptop_llm_tinystories"
MODEL_DIR = "./my_laptop_llm_tinystories_transformer_decoder"

# 2. Create the from-scratch decoder
model = SimpleScratchDecoder(vocab_size=len(tokenizer), d_model=128, num_heads=4)

# To train the Hugging Face GPT-2 model instead, comment out the model line above
# and uncomment this block:
# config = GPT2Config(
#     vocab_size=len(tokenizer),
#     n_positions=256,
#     n_embd=128,
#     n_layer=4,
#     n_head=4,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
# )
# model = GPT2LMHeadModel(config)

# 3. Tokenize TinyStories
def tokenize_function(examples):
    tokens = tokenizer(
        [f"{text}{tokenizer.eos_token}" for text in examples["text"]],
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    tokens["labels"] = [
        [token_id if mask else -100 for token_id, mask in zip(input_ids, attention_mask)]
        for input_ids, attention_mask in zip(tokens["input_ids"], tokens["attention_mask"])
    ]
    return tokens


# Use 10,000 samples for training
small_dataset = raw_dataset.select(range(10000))
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
    data_collator=default_data_collator,
)

trainer.train()

# Save the decoder weights and GPT-2 tokenizer in the same directory
torch.save(model.state_dict(), f"{MODEL_DIR}/model_state.pt")
tokenizer.save_pretrained(MODEL_DIR)
