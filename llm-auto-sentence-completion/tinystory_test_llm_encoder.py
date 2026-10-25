import torch
from transformers import BertForMaskedLM, BertTokenizerFast
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

# 1. Path to your saved encoder model
MODEL_PATH = "./my_laptop_llm_tinystories_encoder"

print(f"Loading model and tokenizer from {MODEL_PATH}...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForMaskedLM.from_pretrained(MODEL_PATH)

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

print("\n================ ENCODER MASKED-WORD TEST RESULTS ================\n")

for prompt in prompts:
    masked_prompt = f"{prompt} {tokenizer.mask_token}"
    inputs = tokenizer(masked_prompt, return_tensors="pt").to(device)
    mask_position = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0]
    print(f"Prompt: {prompt}")

    for settings in generation_settings:
        with torch.no_grad():
            logits = model(**inputs).logits[0, mask_position]

        logits = logits.unsqueeze(0) / settings["temperature"]
        logits = TopKLogitsWarper(top_k=settings["top_k"])(inputs["input_ids"], logits)
        logits = TopPLogitsWarper(top_p=settings["top_p"])(inputs["input_ids"], logits)
        probabilities = torch.softmax(logits[0], dim=-1)
        predicted_id = torch.multinomial(probabilities, num_samples=1).item()
        predicted_word = tokenizer.decode([predicted_id], skip_special_tokens=True)
        generated_text = f"{prompt} {predicted_word}"

        print(f"temperature={settings['temperature']}, top_k={settings['top_k']}, top_p={settings['top_p']}")
        print(f"Result: {generated_text}")
    print("-" * 55)
