import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

MODEL_PATH = "./llm_encoder_decoder_model"
tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_text(prompt, temperature, top_k, top_p):
    inputs = tokenizer(f"continue the lesson: {prompt}", return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
