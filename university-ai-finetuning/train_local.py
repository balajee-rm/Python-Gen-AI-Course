import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 1. Base Model ID
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Initializing 4-bit quantization config...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,  # <-- Changed to float32
    bnb_4bit_use_double_quant=True
)

print("Loading base model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float32,            # <-- Added to prevent BFloat16 fallback
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Load Dataset
print("Loading dataset.json...")
dataset = load_dataset("json", data_files="dataset.json")

def format_prompts(example):
    texts = []
    for inst, inp, out in zip(example["instruction"], example["input"], example["output"]):
        # Fixed typo: changed <|im_endتاز|> to <|im_end|>
        text = f"<|im_start|>system\n{inst}<|im_end|>\n<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
        texts.append(text)
    return {"text": texts}

formatted_dataset = dataset.map(format_prompts, batched=True)

# 4. Training Arguments (Forced clean precision to avoid BFloat16 scaler bug)
training_args = SFTConfig(
    output_dir="./local_output",
    dataset_text_field="text",
    max_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=8,
    fp16=False,     # <-- Explicitly set to False
    bf16=False,     # <-- Explicitly set to False
    logging_steps=1,
    save_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset["train"],
    peft_config=peft_config,
    args=training_args,
)

print("Starting local QLoRA fine-tuning...")
trainer.train()

# 5. Save the trained adapter weights
adapter_path = "./student_advisor_adapter"
trainer.model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"Training complete! Adapter weights successfully saved locally at: {adapter_path}")