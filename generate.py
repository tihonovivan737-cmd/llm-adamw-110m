"""Sample text from a checkpoint. Full-context recomputation, no KV cache."""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llm.model import LanguageModel, ModelConfig
from train import autocast


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prompt', default='The purpose of science is')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if args.temperature <= 0 or args.max_new_tokens < 0 or args.top_k < 0:
        p.error('Require temperature > 0, max-new-tokens >= 0, top-k >= 0')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ModelConfig(**checkpoint['config']['model'])
    model = LanguageModel(cfg).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint.parent / 'tokenizer', local_files_only=True)
    ids = tokenizer(args.prompt, add_special_tokens=False)['input_ids']
    if not ids:
        ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    precision = checkpoint['config']['training']['precision']
    for _ in range(args.max_new_tokens):
        with autocast(device, precision):
            logits = model(tokens[:, -cfg.seq_len:])[:, -1].float() / args.temperature
        if args.top_k:
            boundary = logits.topk(min(args.top_k, cfg.vocab_size)).values[:, -1:]
            logits.masked_fill_(logits < boundary, -float('inf'))
        token = torch.multinomial(logits.softmax(-1), 1)
        tokens = torch.cat((tokens, token), dim=1)
        if token.item() == tokenizer.eos_token_id:
            break
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print(tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True))


if __name__ == '__main__':
    main()
