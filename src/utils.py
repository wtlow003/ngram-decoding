import numpy as np
import torch


# adapted from: https://github.com/apoorvumang/prompt-lookup-decoding?tab=readme-ov-file
def generate_candidate_tokens(
    input_ids: torch.Tensor, n_grams: torch.Tensor, ngrams_size: int, K: int
):
    # unfold the tensor into windows of `pattern_len + following_elements_count`
    window = input_ids.unfold(dimension=1, size=ngrams_size, step=1)
    # compare each window with the pattern (only the parts corresponding to the pattern)
    matching_window_indices = (window == n_grams).all(dim=2)
    # extract the indices where there are matches
    matching_indices = matching_window_indices.nonzero(as_tuple=True)[1]

    # find candidates with the longest length
    # based on: https://arxiv.org/pdf/2304.04487
    # we choose the candidate with the longest length at random if there are multiple candidates
    candidates = []
    max_length = K
    for idx in matching_indices:
        start_idx = idx + ngrams_size
        end_idx = start_idx + K
        candidate = input_ids[0, start_idx : min(end_idx, input_ids.size(1))]
        length = len(candidate)

        if length == max_length:
            candidates.append(candidate)
        else:
            # we do not consider prefix with no candidates
            if length > max_length:
                max_length = length
                candidates = [candidate]

    if candidates:
        chosen_candidate = candidates[np.random.randint(len(candidates))]
    else:
        chosen_candidate = torch.tensor([], dtype=torch.long, device=input_ids.device)

    return chosen_candidate.unsqueeze(dim=0)
