import torch
from transformers import BertForMaskedLM, BertTokenizerFast
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

MODEL_PATH = "./llm_encoder_model"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForMaskedLM.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_text(prompt, temperature, top_k, top_p):
    inputs = tokenizer(f"{prompt} {tokenizer.mask_token}", return_tensors="pt").to(device)
    mask_position = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0]

    with torch.inference_mode():
        logits = model(**inputs).logits[0, mask_position].unsqueeze(0) / temperature

    logits = TopKLogitsWarper(top_k=top_k)(inputs["input_ids"], logits)
    logits = TopPLogitsWarper(top_p=top_p)(inputs["input_ids"], logits)
    probabilities = torch.softmax(logits[0], dim=-1)
    predicted_id = torch.multinomial(probabilities, num_samples=1).item()
    predicted_word = tokenizer.decode([predicted_id], skip_special_tokens=True)
    return f"{prompt} {predicted_word}".strip()
