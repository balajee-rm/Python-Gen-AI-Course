import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GenerationConfig

MODEL_PATH = "./my_laptop_llm2"

# 1. Load model & tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Clean up BPE tokenization warning
tokenizer.clean_up_tokenization_spaces = False
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. Configure Generation parameters explicitly
gen_config = GenerationConfig(
    max_new_tokens=30,
    do_sample=True,
    top_k=40,
    top_p=0.9,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)

prompts = ["The river is", "The city of", "In the year"]

print("\n================ GENERATION TEST RESULTS ================\n")

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate directly through model using explicitly defined config
    with torch.no_grad():
        output_tokens = model.generate(**inputs, generation_config=gen_config)
    
    text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # Clean WikiText hyphen artifacts
    cleaned_text = text.replace(" @-@ ", "-").replace("@-@", "-")
    
    print(f"Prompt: {prompt}")
    print(f"Result: {cleaned_text}")
    print("-" * 55)