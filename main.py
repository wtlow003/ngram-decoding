import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.greedy_decoding import greedy_decoding
from src.ngram_decoding import ngram_decoding


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="mps",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="mps",
    )
    tokenizer.eos_token_id = 128009

    input_str = """import numpy as np
    import matplotlib.pyplot as plt

    # Calculate the average
    average_throughput = np.mean(tokens_per_sec_arr)
    print(f"Average Throughput: {average_throughput} tokens/sec")

    # Plotting the histogram
    plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Throughput Values')
    plt.xlabel('Tokens per Second')
    plt.ylabel('Frequency')
    plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
    plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
    plt.show()
    """
    question = "Can you please change x axis to start from 0"
    prompt = "<|start_header_id|>user<|end_header_id|>\nCode:```python\n{code_text}``` \n\n Question: {question} \n\n Modified code:\n<|start_header_id|>assistant<|end_header_id|>".format(
        code_text=input_str, question=question
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    # do a simple forward pass to warm up related kernel
    model(input_ids)

    print("\nNgram Decoding:")
    nd_start = time.perf_counter()
    # output = ngram_decoding(input_ids, model, ngrams_size=3, K=10, n=400)
    nd_output_ids = []
    for token_id, speculated in ngram_decoding(
        input_ids, model, tokenizer, ngrams_size=3, K=10, n=400
    ):
        nd_output_ids.append(token_id)
        if speculated:
            print(f"\033[92m{tokenizer.decode(token_id)}\033[0m", end="", flush=True)
        else:
            print(
                tokenizer.decode(token_id, skip_special_tokens=True), end="", flush=True
            )
    nd_end = time.perf_counter()
    nd_time = nd_end - nd_start
    print(
        f"\nTime taken: {nd_end - nd_start} seconds, {len(nd_output_ids) / nd_time} tokens/s"
    )

    print("\nGreedy Decoding:")
    gd_start = time.perf_counter()
    gd_output_ids = []
    for token_id in greedy_decoding(input_ids, model, tokenizer, n=400):
        gd_output_ids.append(token_id)
        print(tokenizer.decode(token_id, skip_special_tokens=True), end="", flush=True)
    gd_end = time.perf_counter()
    gd_time = gd_end - gd_start
    print(
        f"\nTime taken: {gd_end - gd_start} seconds, {len(gd_output_ids) / gd_time} tokens/s"
    )

    # calculate speedup
    print("\n")
    speedup = gd_time / nd_time
    speedup_percentage = (speedup - 1) * 100
    print(f"Speedup: {speedup:.2f}x")
    print(f"Percentage improvement: {speedup_percentage:.2f}%")


if __name__ == "__main__":
    main()
