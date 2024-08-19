import torch
from transformers import PreTrainedTokenizer


@torch.no_grad()
def greedy_decoding(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    n: int = 400,
):
    eos_token_id = tokenizer.eos_token_id

    seq_len = input_ids.shape[1]
    T = seq_len + n

    while input_ids.shape[1] < T:
        logits = model(input_ids).logits
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(dim=1)], dim=1)
        yield next_token_id.item()
        if next_token_id == eos_token_id:
            break

    return input_ids
