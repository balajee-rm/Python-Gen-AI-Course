from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
)

# 1. Tokenizer & Blank Model Setup
tokenizer = T5TokenizerFast.from_pretrained("t5-small")

config = T5Config(
    vocab_size=len(tokenizer),
    d_model=128,
    d_kv=32,
    d_ff=512,
    num_layers=4,
    num_decoder_layers=4,
    num_heads=4,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,
)
model = T5ForConditionalGeneration(config)

# 2. Load TinyStories (Clean, simple dataset for small models)
print("Loading TinyStories dataset...")
raw_dataset = load_dataset("roneneldan/TinyStories", split="train")


def tokenize_function(examples):
    inputs = []
    targets = []

    for text in examples["text"]:
        text = text.strip()
        split_index = text.rfind(" ", 0, len(text) // 2)
        if split_index <= 0:
            split_index = text.find(" ", len(text) // 2)
        if split_index <= 0 or split_index >= len(text) - 1:
            split_index = max(1, len(text) // 2)

        inputs.append(f"continue the story: {text[:split_index].strip()}")
        targets.append(text[split_index:].strip())

    return tokenizer(
        inputs,
        text_target=targets,
        truncation=True,
        max_length=128,
    )


# Use 10,000 samples for better learning
small_dataset = raw_dataset.select(range(10000))
tokenized_dataset = small_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Training
training_args = TrainingArguments(
    output_dir="./my_laptop_llm_tinystories_encoder_decoder",
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
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
)

trainer.train()

model.save_pretrained("./my_laptop_llm_tinystories_encoder_decoder")
tokenizer.save_pretrained("./my_laptop_llm_tinystories_encoder_decoder")
