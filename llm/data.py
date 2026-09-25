import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def document_key(text):
    return hashlib.sha256(text.strip().encode('utf-8')).digest()


def validation_document(key, fraction, seed):
    digest = hashlib.sha256(str(seed).encode() + key).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64 < fraction


class TokenFile:
    def __init__(self, directory, split):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        self.info = self.manifest['splits'][split]
        path = self.directory / f'{split}.bin'
        dtype = np.dtype(self.manifest['dtype'])
        if path.stat().st_size != self.info['stored_tokens'] * dtype.itemsize:
            raise ValueError(f'Incomplete/corrupt token file: {path}')
        self.tokens = np.memmap(path, mode='r', dtype=dtype)
        self.target_tokens = len(self.tokens) - 1

    def batch(self, offset, count, batch_size, seq_len, device):
        """Each label is used once, even in the final partially filled microbatch."""
        capacity = batch_size * seq_len
        if not 0 <= count <= capacity or offset < 0 or (count and offset + count > self.target_tokens):
            raise ValueError('Invalid token slice')
        x = np.zeros(capacity, dtype=np.int64)
        y = np.full(capacity, -100, dtype=np.int64)
        if count:
            x[:count] = self.tokens[offset:offset + count]
            y[:count] = self.tokens[offset + 1:offset + count + 1]
        return (torch.from_numpy(x.reshape(batch_size, seq_len)).to(device),
                torch.from_numpy(y.reshape(batch_size, seq_len)).to(device))


def learning_rate(tokens_after_step, cfg):
    import math
    total = cfg['max_tokens']
    warmup = max(1, int(total * cfg['warmup_fraction']))
    peak = cfg['learning_rate']
    if tokens_after_step <= warmup:
        return peak * tokens_after_step / warmup
    progress = min(1.0, (tokens_after_step - warmup) / max(1, total - warmup))
    return peak * (cfg['min_lr_ratio'] + (1 - cfg['min_lr_ratio']) *
                   0.5 * (1 + math.cos(math.pi * progress)))
