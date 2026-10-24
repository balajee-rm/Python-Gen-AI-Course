import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline

MODEL_PATH = "./my_laptop_llm"

# ---------------------------------------------------------
# 1. Load Trained Model & Tokenizer
# ---------------------------------------------------------
print("Loading model and tokenizer from disk...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Ensure tokenizer pad token is defined
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---------------------------------------------------------
# 2. Build Generation Pipeline
# ---------------------------------------------------------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# ---------------------------------------------------------
# 3. Test Prompts
# ---------------------------------------------------------
test_prompts = [
    "Python is",
    "Artificial intelligence",
    "Vision language models",
    "The river is"
]

print("\n================ GENERATION TEST RESULTS ================\n")

for prompt in test_prompts:
    output = generator(
        prompt,
        max_new_tokens=25,               # Stay well within the 256-token limit
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,                  # Enable sampling for varied outputs
        top_k=50,                        # Sample top 50 tokens
        top_p=0.95,                      # Nucleus sampling
        temperature=0.7,                 # Control output randomness
    )
    
    generated_text = output[0]["generated_text"]
    print(f"Prompt: {prompt}")
    print(f"Result: {generated_text}")
    print("-" * 55)