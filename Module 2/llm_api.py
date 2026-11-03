import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from test_llm_decoder_transformer import generate_text as generate_scratch_decoder_text
from test_llm_decoder import generate_text
from test_llm_encoder import generate_text as generate_encoder_text
from test_llm_encoder_decoder import generate_text as generate_encoder_decoder_text

app = FastAPI(title="LLM Auto Sentence Completion API", version="1.0")


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)
    temperature: float = Field(default=0.7, gt=0)
    top_k: int = Field(default=50, ge=1)
    top_p: float = Field(default=0.9, gt=0, le=1)


@app.post("/decoder/generate")
def generate(request: GenerationRequest):
    return {
        "prompt": request.prompt,
        "output": generate_text(
            request.prompt,
            request.temperature,
            request.top_k,
            request.top_p,
        ),
    }


@app.post("/encoder/generate")
def generate_encoder(request: GenerationRequest):
    return {
        "prompt": request.prompt,
        "output": generate_encoder_text(
            request.prompt,
            request.temperature,
            request.top_k,
            request.top_p,
        ),
    }


@app.post("/encoder-decoder/generate")
def generate_encoder_decoder(request: GenerationRequest):
    return {
        "prompt": request.prompt,
        "output": generate_encoder_decoder_text(
            request.prompt,
            request.temperature,
            request.top_k,
            request.top_p,
        ),
    }


@app.post("/scratch-decoder/generate")
def generate_scratch_decoder(request: GenerationRequest):
    return {
        "prompt": request.prompt,
        "output": generate_scratch_decoder_text(
            request.prompt,
            request.temperature,
            request.top_k,
            request.top_p,
        ),
    }
