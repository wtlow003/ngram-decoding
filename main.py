import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.greedy_decoding import greedy_decoding
from src.ngram_decoding import ngram_decoding

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def main(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map=DEVICE,
        use_cache=False,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    tokenizer.eos_token_id = 128009

    input_str = """
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
    """
    question = "Can you the variable name 'candidates' to 'candidates_tokens'?"
    prompt = "<|start_header_id|>user<|end_header_id|>\nCode:\n```python{code_text}``` \n\n Question: {question} \n\n Modified code:\n<|start_header_id|>assistant<|end_header_id|>".format(
        code_text=input_str, question=question
    )
    print(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    if args.decoding_method == "ngram":
        # warm-up run
        print("Starting warm-up run")
        ngram_decoding(input_ids, model, tokenizer, ngrams_size=3, K=10, n=50)
        print("Warm-up complete.")

        # actual run
        print("\nNgram Decoding:")
        torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()
        nd_start = time.perf_counter()
        nd_output_ids = []
        for token_id, speculated in ngram_decoding(
            input_ids, model, tokenizer, ngrams_size=3, K=10, n=400
        ):
            nd_output_ids.append(token_id)
            if speculated:
                print(
                    f"\033[92m{tokenizer.decode(token_id)}\033[0m", end="", flush=True
                )
            else:
                print(
                    tokenizer.decode(token_id, skip_special_tokens=True),
                    end="",
                    flush=True,
                )
        torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()
        nd_end = time.perf_counter()
        nd_time = nd_end - nd_start
        print(
            f"\nTime taken: {nd_end - nd_start} seconds, {len(nd_output_ids) / nd_time} tokens/s"
        )
    else:
        # warm-up run
        print("Starting warm-up run")
        greedy_decoding(input_ids, model, tokenizer, n=50)
        print("Warm-up complete.")

        print("\nGreedy Decoding:")
        torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()
        gd_start = time.perf_counter()
        gd_output_ids = []
        for token_id in greedy_decoding(input_ids, model, tokenizer, n=400):
            gd_output_ids.append(token_id)
            print(
                tokenizer.decode(token_id, skip_special_tokens=True), end="", flush=True
            )
        torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()
        gd_end = time.perf_counter()
        gd_time = gd_end - gd_start
        print(
            f"\nTime taken: {gd_end - gd_start} seconds, {len(gd_output_ids) / gd_time} tokens/s"
        )

    gc.collect()
    torch.cuda.empty_cache() if DEVICE == "cuda" else torch.mps.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--decoding-method", type=str, required=True, choices=["greedy", "ngram"]
    )
    args = parser.parse_args()
    main(args)
