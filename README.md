<h1 align="center">N-grams Decodinge</h1>

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