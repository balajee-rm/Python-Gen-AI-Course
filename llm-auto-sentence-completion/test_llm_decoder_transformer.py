import torch
from transformers import GPT2TokenizerFast

from transformer_decoder import SimpleScratchDecoder

MODEL_PATH = "./llm_decoder_transformer_model"
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
model = SimpleScratchDecoder(vocab_size=len(tokenizer), d_model=128, num_heads=4)
model.load_state_dict(
    torch.load(f"{MODEL_PATH}/model_state.pt", map_location="cpu", weights_only=True)
)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_text(prompt, temperature, top_k, top_p):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=40,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
