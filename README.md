# LLM Fine-Tuning Examples (SFT, LoRA, QLoRA)

This repository contains clear, runnable examples of how to fine-tune open-weight large language models (LLMs) using three popular techniques:

**Supervised Fine-Tuning (SFT)**  
**Parameter-Efficient Fine-Tuning (LoRA)**  
**Quantized LoRA (QLoRA)**

All code uses [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) and [PEFT](https://huggingface.co/docs/peft/index).

---

## What is included?

- Minimal, readable `main.py` for training with SFT, LoRA, or QLoRA
- Configurable model loading (e.g. Qwen 1.5 1.8B)
- Simple JSONL dataset loading
- Tokenization and preprocessing
- Command-line arguments with `argparse`
- Example requirements.txt

---

## Repository Structure
```bash
.
├── dataset
├──── train_dataset.jsonl
├── main.py
├── config.py
├── data.py
├── model.py
├── train.py
├── eval.py
├── requirements.txt
└── README.md
```
---

## Requirements

- Python 3.8+
- PyTorch
- GPU recommended (but can also run CPU with smaller models)

Example `requirements.txt`:
- torch
- transformers
- datasets
- peft
- accelerate
- bitsandbytes


---

## Features
Features
 - SFT: Standard supervised fine-tuning
 - LoRA: Parameter-efficient adapters for smaller, faster training
 - QLoRA: Memory-efficient 4-bit quantization with LoRA

## Configuration
You can edit config.py to set:
 - Model name
 - Batch size
 - Epochs
 - Learning rate
 - LoRA settings (rank, alpha, dropout, target modules)

## References
 - [Hugging Face Transformers](https://huggingface.co/docs/transformers)
 - [Hugging Face PEFT](https://huggingface.co/docs/peft)
 - [Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv preprint arXiv:2305.14314.](https://arxiv.org/abs/2305.14314)
 - [Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.](https://arxiv.org/abs/2106.09685)

