<h1 align="center">N-grams Decoding</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.9.10-orange"
         alt="python version">
     <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"
          alt="uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json"
         alt="ruff">
</p>

## About

This repository contains the implementation of the ngram-decoding (aka *prompt lookup decoding*) method for faster LLM inference.

This exploration aims to understand the using n-grams for loseless accelaration of LLM inference, as proposed in: 

1. [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding?tab=readme-ov-file)
2. [LLMA Decoding](https://github.com/microsoft/LMOps/tree/main/llma)

Combining the core ideas from both methods, I explored the following algorithm built upon the aforementioned works:

1. Match the n-grams in the prompt with the tokens in the input sequence, and obtain `K` candidate tokens.
2. If multiple candidates are found, select the set with the most candidate tokens. In case of a tie, a random selection is made.
3. If no candidate tokens are identified, default to single-step greedy decoding.

> [!NOTE]
The number of tokens generated per step in n-gram decoding ranges from `1` to `K+1`.

4. Repeat the above steps until either the maximum `n` number of tokens is reached or the `EOS` (e.g., `<|eot_id|>`) token is generated.

## Getting Started

This project uses uv for dependency management. To install UV, run the following command:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
pip install uv

# With pipx.
pipx install uv

# With Homebrew.
brew install uv

# With Pacman.
pacman -S uv
```

Thereafter, install the rest of the dependencies using uv:

```bash
# create a virtual env
uv venv

# install dependencies
uv pip install -r requirements.txt  # Install from a requirements.txt file.
```

## Usage

> [!NOTE]
>
> Currently, the script only supports `Meta-Llama-3.1-8B-Instruct` model.

```bash
# check cli options
python main.py --help

usage: main.py [-h] [--model MODEL] --decoding-method {greedy,ngram}

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --decoding-method {greedy,ngram}
```

Running LLM inference comparison script:

```bash
# ngram decoding
python main.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --decoding-method ngram

# greedy decoding
python main.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --decoding-method greedy
```

## Results

The following results are obtained on `A100` GPU with `40GB` RAM, with the following settings:

1. `ngrams_size` = 3
2. `K` = 10
3. `n` = 400

https://github.com/user-attachments/assets/5b103571-a9ea-4e46-ad52-c3f91589c83e

Using the following example prompt:

```
<|start_header_id|>user<|end_header_id|>
Code:
```python
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
    ``` 

 Question: Can you the variable name 'candidates' to 'candidates_tokens'? 

 Modified code:
<|start_header_id|>assistant<|end_header_id|>
```

The following timings are observed:

|    Decoding Method   |  Time Taken (s)  |  Token/secs  |   Speedup   |
| -------------------- | ---------------- | ------------ | ----------- |
|    Greedy Decoding   |      26.4        |     14.0     |      1x     | 
|    Ngrams Decoding   |      12.8        |     28.9     |     ~2x     | 

In the simple demonstration experiment, we achieved results comparable to those of the original [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding?tab=readme-ov-file) implementation and the figures reported in [LLMA Decoding](https://github.com/microsoft/LMOps/tree/main/llma). Both decoding methods demonstrated approximately a 2-3x improvement in speed over greedy decoding.

## References

```
@misc{saxena2023prompt,
    title = {Prompt Lookup Decoding},
    author = {Apoorv Saxena},
    year = {2023},
    month = {November},
    url = {https://github.com/apoorvumang/prompt-lookup-decoding/}
}

@misc{yang2023inferencereferencelosslessacceleration,
      title={Inference with Reference: Lossless Acceleration of Large Language Models}, 
      author={Nan Yang and Tao Ge and Liang Wang and Binxing Jiao and Daxin Jiang and Linjun Yang and Rangan Majumder and Furu Wei},
      year={2023},
      eprint={2304.04487},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2304.04487}, 
}
```

## Acknowledgements

The implementation for ngram-decoding is build upon the following repository:

1. https://github.com/apoorvumang/prompt-lookup-decoding?tab=readme-ov-file
2. https://github.com/microsoft/LMOps/tree/main/llma
