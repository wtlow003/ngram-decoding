from typing import Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .utils import generate_candidate_tokens


@torch.no_grad()
def ngram_decoding(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ngrams_size: int,
    K: int,
    n: int,
):
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    eos_token_id_tensor = torch.tensor(
        [eos_token_id], dtype=torch.long, device=input_ids.device
    )
    seq_len = input_ids.shape[1]
    T = seq_len + n

    while input_ids.shape[1] < T:
        prefix = input_ids
        cur_len = input_ids.shape[1]

        # -----------------------------------------
        # Step 1: Generate N-grams
        # -----------------------------------------

        n_grams = input_ids[0, -ngrams_size:]

        # -----------------------------------------
        # Step 2: Generate K candidates tokens using the N-grams
        # -----------------------------------------

        candidate_tokens = generate_candidate_tokens(input_ids, n_grams, ngrams_size, K)

        # -----------------------------------------
        # Step 3: Validate the candidates using the LLM
        # -----------------------------------------

        # based on: https://arxiv.org/pdf/2304.04487
        # if we did not find any candidates tokens, we default to single-step decoding
        if candidate_tokens.shape[1] == 0:
            logits = model(input_ids).logits[:, -1, :]
            next_token = logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(dim=0)], dim=1)
            yield (next_token.item(), False)
            if next_token.item() == eos_token_id:
                break
            continue

        prefix = torch.cat([input_ids, candidate_tokens], dim=1)
        # include the max(1, K) + 1 in the logits
        logits = model(prefix).logits[:, -candidate_tokens.shape[1] - 1 :, :]

        assert (
            logits.shape[1] == candidate_tokens.shape[1] + 1
        ), f"Expected logits shape: {candidate_tokens.shape[1] + 1}, got: {logits.shape[1]}"

        selected_tokens = logits.argmax(dim=-1)
        # calculate the number of consecutive matching tokens between candidate_tokens and selected_tokens:
        # 1. Compare candidate_tokens with selected_tokens
        # 2. Invert the comparison result
        # 3. Calculate cumulative sum of mismatches
        # 4. Create a mask for positions before the first mismatch
        # 5. Sum up the mask to get the count of consecutive matches
        n_matches = (
            (~(candidate_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1
        ).sum()
        # minus 1 to prevent overshooting max allowable tokens
        n_matches = min(n_matches, T - cur_len - 1)

        # this allow us to generate at least 1 token or K+1 tokens max
        valid_tokens = selected_tokens[:, : n_matches + 1]
        # print("selected from prompt: ", tokenizer.decode(valid_tokens[0]))
        for token_id in valid_tokens[0]:
            yield (token_id.item(), True)
        input_ids = torch.cat([input_ids, valid_tokens], dim=1)

        if input_ids.shape[1] >= T:  # Check if we've reached the desired length
            break

        if (valid_tokens == eos_token_id_tensor.item()).any():
            break

    return input_ids
