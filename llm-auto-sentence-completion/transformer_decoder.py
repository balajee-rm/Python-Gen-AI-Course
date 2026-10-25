import math
import torch
import torch.nn as nn


# ==========================================
# 1. Simple From-Scratch Decoder Architecture
# ==========================================

class SimpleScratchDecoder(nn.Module):
    def __init__(self, vocab_size=8000, d_model=128, num_heads=4):
        super().__init__()
        
        # Embeddings (Words + Positions)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(256, d_model)
        
        # Attention Projections (Q, K, V, Out)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Final Output Layer (Maps vector back to vocabulary scores)
        self.to_vocab = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Create embeddings + positions
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        # Step 2: Compute Query, Key, Value
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Scaled Dot-Product Attention Scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Step 4: Causal Mask (Stops the model from looking into the future)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        
        # Step 5: Softmax & Attention Output
        attn_weights = torch.softmax(scores, dim=-1)
        context = (attn_weights @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Step 6: Final Projection & Vocabulary Logits
        output = self.W_o(context)
        logits = self.to_vocab(output)

        if labels is None:
            return logits

        loss = nn.functional.cross_entropy(
            logits[:, :-1].transpose(1, 2),
            labels[:, 1:],
            ignore_index=-100,
        )
        return loss, logits

    # Simple, non-complex generation method
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=40, temperature=1.0, **kwargs):
        for _ in range(max_new_tokens):
            logits = self(input_ids)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids