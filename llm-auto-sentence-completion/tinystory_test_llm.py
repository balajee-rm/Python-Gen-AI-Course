import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 1. Path to your saved model directory
MODEL_PATH = "./my_laptop_llm_tinystories"

print(f"Loading model and tokenizer from {MODEL_PATH}...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Set model to evaluation mode
model.eval()

# Check device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Test Prompts
prompts = [
    "Hi i am",
    "In the college",
    "the river,"
    "i am reading the test book in science about the earch and interested in space science with"
]

print("\n================ GENERATION TEST RESULTS ================\n")

for prompt in prompts:
    # Tokenize input text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text using top-k and top-p sampling to prevent repetitive looping
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,       # Number of new tokens to generate
            do_sample=True,          # Enable sampling for variety
            temperature=0.7,         # Controls randomness (lower = more deterministic)
            top_k=50,                # Keeps top 50 most likely tokens
            top_p=0.9,               # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and print generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Result: {generated_text}")
    print("-" * 55)