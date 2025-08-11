# GPT-2 From Scratch

This repository contains my from-scratch implementation of a GPT-2-like language model, inspired by the architecture of OpenAI's GPT-2. The project demonstrates my deep learning engineering skills, including model design, training loop construction, and distributed training support.

## Project Motivation

I built this project to deepen my understanding of transformer architectures and large language model training. The codebase is designed to be educational, modular, and easy to follow, making it a great resource for anyone interested in how modern language models are constructed and trained.

## Features
- Full GPT-2 (124M) architecture implementation in PyTorch
- Training loop with distributed data parallel (DDP) support
- Data loading pipeline for large-scale text datasets
- Validation and sample text generation
- HellaSwag evaluation integration

## Example Output
After training, the model can generate text like:
```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
```

## Contact
For questions, collaborations, or more information, please contact me:
- Name: Pranav Pattanshetti
- Email: [your-email@example.com]
- LinkedIn: [your-linkedin-profile]

## License
MIT

## Technologies Used
- Python 3
- PyTorch
- NumPy
- tiktoken
- HuggingFace Transformers (for optional pretrained weights)

## How to Run
1. Install dependencies:
   ```bash
   pip install torch numpy tiktoken transformers
   ```
2. Prepare your dataset in the expected format (see code for details).
3. Run the training script:
   ```bash
   python train_gpt2.py
   ```

For distributed training, use:
```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```
