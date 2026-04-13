from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from addition.config import ExperimentConfig


@dataclass
class ModelOutput:
    digit_logits: torch.Tensor
    carry_logits: torch.Tensor
    final_hidden: torch.Tensor
    readout_hidden: torch.Tensor
    latent_history: list[torch.Tensor]
    attention_weights: torch.Tensor | None


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor, need_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq_len = hidden_states.shape[1]
        causal_mask = torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool).triu(1)
        normed = self.ln_1(hidden_states)
        attn_output, attn_weights = self.attn(
            normed,
            normed,
            normed,
            need_weights=need_weights,
            average_attn_weights=False,
            attn_mask=causal_mask,
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, attn_weights if need_weights else None


class AdditionTransformer(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.discrete_vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.d_model)
        self.latent_type_embedding = nn.Parameter(torch.zeros(config.d_model))
        self.block = TransformerBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.digit_head = nn.Linear(config.d_model, 10)
        self.carry_head = nn.Linear(config.d_model, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_type_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.digit_head.weight)
        nn.init.zeros_(self.digit_head.bias)
        nn.init.xavier_uniform_(self.carry_head.weight)
        nn.init.zeros_(self.carry_head.bias)

    def embed_discrete_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.token_embedding(input_ids) + self.position_embedding(positions)

    def _run_block(
        self,
        embeddings: torch.Tensor,
        *,
        need_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states, attention_weights = self.block(embeddings, need_weights=need_attention)
        hidden_states = self.final_ln(hidden_states)
        return hidden_states, attention_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        latent_steps: int = 0,
        return_attention: bool = False,
    ) -> ModelOutput:
        base_embeddings = self.embed_discrete_tokens(input_ids)
        latent_history: list[torch.Tensor] = []
        attention_weights: torch.Tensor | None = None

        hidden_states, attention_weights = self._run_block(base_embeddings, need_attention=return_attention)
        readout_hidden = hidden_states[:, -1, :]
        latent_history.append(readout_hidden)

        latent_embeddings: list[torch.Tensor] = []
        for step_index in range(int(latent_steps)):
            latent_token = readout_hidden.unsqueeze(1) + self.latent_type_embedding.view(1, 1, -1)
            latent_position_index = base_embeddings.shape[1] + step_index
            latent_token = latent_token + self.position_embedding.weight[latent_position_index].view(1, 1, -1)
            latent_embeddings.append(latent_token)
            hidden_states, attention_weights = self._run_block(
                torch.cat([base_embeddings] + latent_embeddings, dim=1),
                need_attention=return_attention,
            )
            readout_hidden = hidden_states[:, -1, :]
            latent_history.append(readout_hidden)

        final_hidden = readout_hidden
        digit_logits = self.digit_head(final_hidden)
        carry_logits = self.carry_head(final_hidden)
        return ModelOutput(
            digit_logits=digit_logits,
            carry_logits=carry_logits,
            final_hidden=final_hidden,
            readout_hidden=readout_hidden,
            latent_history=latent_history,
            attention_weights=attention_weights,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(config: ExperimentConfig, device: str | None = None) -> AdditionTransformer:
    model = AdditionTransformer(config)
    if device is not None:
        model = model.to(device)
    return model


@torch.no_grad()
def describe_model(config: ExperimentConfig) -> dict[str, int]:
    model = build_model(config)
    total_params = model.parameter_count()
    head_params = sum(parameter.numel() for name, parameter in model.named_parameters() if "head" in name)
    embedding_params = sum(parameter.numel() for name, parameter in model.named_parameters() if "embedding" in name)
    return {
        "total_params": int(total_params),
        "embedding_params": int(embedding_params),
        "head_params": int(head_params),
        "backbone_params": int(total_params - head_params),
    }
