import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
# Uncomment to test the custom scratch decoder:
from transformer_decoder import SimpleScratchDecoder

# 1. Path to your saved decoder model
# MODEL_PATH = "./my_laptop_llm_tinystories"
# For the scratch decoder, comment the GPT-2 path above and uncomment below 1 line:
MODEL_PATH = "./my_laptop_llm_tinystories_transformer_decoder"

print(f"Loading model and tokenizer from {MODEL_PATH}...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)

# model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
# For the scratch decoder, comment the GPT-2 model line above and uncomment below 2 lines:
model = SimpleScratchDecoder(vocab_size=len(tokenizer), d_model=128, num_heads=4)
model.load_state_dict(torch.load(f"{MODEL_PATH}/model_state.pt", map_location="cpu", weights_only=True))

# Set model to evaluation mode
model.eval()

# Check device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Test Prompts
prompts = [
    "Hi i am",
    "In the college",
]

generation_settings = [
    {"temperature": 0.3, "top_k": 10, "top_p": 0.8},
    {"temperature": 0.7, "top_k": 50, "top_p": 0.9},
    {"temperature": 1.0, "top_k": 100, "top_p": 0.95},
]

print("\n================ DECODER GENERATION TEST RESULTS ================\n")

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Prompt: {prompt}")

    for settings in generation_settings:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                **settings,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"temperature={settings['temperature']}, top_k={settings['top_k']}, top_p={settings['top_p']}")
        print(f"Result: {generated_text}")
    print("-" * 55)
