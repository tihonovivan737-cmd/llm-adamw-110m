"""Stream a weighted mixture of pinned text datasets into exact-budget token files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from llm.data import document_key, validation_document


def mixed_rows(source_configs, seed, shuffle_buffer):
    """Yield unique source streams in weighted document-level proportions."""
    active = []
    for i, source in enumerate(source_configs):
        kwargs = {
            'path': source['dataset'],
            'split': source.get('split', 'train'),
            'streaming': True,
            'revision': source['revision'],
        }
        if source.get('subset'):
            kwargs['name'] = source['subset']
        stream = load_dataset(**kwargs).select_columns(['text'])
        stream = stream.shuffle(seed=seed + i, buffer_size=shuffle_buffer)
        source_name = source['name']
        stream = stream.map(lambda row, name=source_name: {'source': name})
        active.append({'name': source_name, 'weight': source['weight'], 'rows': iter(stream)})

    rng = np.random.default_rng(seed)
    while active:
        weights = np.asarray([source['weight'] for source in active], dtype=np.float64)
        weights /= weights.sum()
        index = int(rng.choice(len(active), p=weights))
        try:
            yield next(active[index]['rows'])
        except StopIteration:
            active.pop(index)


def prepare(args):
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    cfg = config['data']
    train_tokens = args.train_tokens if args.train_tokens is not None else cfg['train_tokens']
    val_tokens = args.val_tokens if args.val_tokens is not None else cfg['val_tokens']
    sources = cfg['sources']
    if (min(train_tokens, val_tokens) < 1 or not 0 < cfg['val_fraction'] < 1
            or not sources or any(source['weight'] <= 0 for source in sources)):
        raise ValueError('Positive token budgets and a validation fraction in (0,1) are required')
    if len({source['name'] for source in sources}) != len(sources):
        raise ValueError('Dataset source names must be unique')
    out = Path(args.output)
    temp = out.with_name(out.name + '.partial')
    if out.exists() or temp.exists():
        raise FileExistsError(f'{out} or {temp} exists. Use a new output directory; no files were overwritten.')
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer'], revision=cfg['tokenizer_revision'],
                                               use_fast=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError(f'Tokenizer {cfg["tokenizer"]} has no EOS token configured')
    if len(tokenizer) != config['model']['vocab_size']:
        raise ValueError(
            f'Tokenizer vocabulary mismatch: len(tokenizer)={len(tokenizer)}, '
            f'model.vocab_size={config["model"]["vocab_size"]}'
        )
    dtype = '<u2' if len(tokenizer) <= 65536 else '<u4'
    temp.mkdir(parents=True)
    tokenizer.save_pretrained(temp / 'tokenizer')
    print(f'Tokenizer: {cfg["tokenizer"]}, vocabulary={len(tokenizer)}, EOS={tokenizer.eos_token_id}', flush=True)
    dataset = mixed_rows(sources, cfg['seed'], cfg['shuffle_buffer'])
    goals = {'train': train_tokens + 1, 'val': val_tokens + 1}
    counts = dict.fromkeys(goals, 0)
    docs = dict.fromkeys(goals, 0)
    source_counts = {source['name']: {'documents': 0, 'train_tokens': 0, 'val_tokens': 0}
                     for source in sources}
    hashes = {s: hashlib.sha256() for s in goals}
    files = {s: (temp / f'{s}.bin').open('wb') for s in goals}
    db = sqlite3.connect(temp / 'seen.sqlite')
    db.execute('CREATE TABLE seen (hash BLOB PRIMARY KEY) WITHOUT ROWID')
    seen = duplicates = 0
    last_log = time.monotonic()
    try:
        texts, splits, source_names = [], [], []

        def flush():
            if not texts:
                return
            batches = tokenizer(texts, add_special_tokens=False, truncation=False,
                                return_attention_mask=False)['input_ids']
            for ids, split, source_name in zip(batches, splits, source_names):
                available = goals[split] - counts[split]
                if available <= 0:
                    continue
                ids.append(tokenizer.eos_token_id)
                chunk = np.asarray(ids[:available], dtype=dtype).tobytes()
                files[split].write(chunk)
                hashes[split].update(chunk)
                token_count = len(chunk) // np.dtype(dtype).itemsize
                counts[split] += token_count
                docs[split] += 1
                source_counts[source_name]['documents'] += 1
                source_counts[source_name][f'{split}_tokens'] += token_count
            texts.clear()
            splits.clear()
            source_names.clear()
            db.commit()

        for row in dataset:
            text = row.get('text', '').strip()
            if not text:
                continue
            seen += 1
            key = document_key(text)
            split = 'val' if validation_document(key, cfg['val_fraction'], cfg['seed']) else 'train'
            if counts[split] >= goals[split]:
                continue
            if db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (key,)).rowcount == 0:
                duplicates += 1
                continue
            texts.append(text)
            splits.append(split)
            source_names.append(row['source'])
            if len(texts) >= 128:
                flush()
                if time.monotonic() - last_log > 5:
                    print(f'train={counts["train"]:,}/{goals["train"]:,}; val={counts["val"]:,}/{goals["val"]:,}', flush=True)
                    last_log = time.monotonic()
                if all(counts[s] == goals[s] for s in goals):
                    break
        flush()
        if any(counts[s] != goals[s] for s in goals):
            raise RuntimeError(f'Source exhausted before reaching token budgets: {counts}')
    finally:
        for f in files.values():
            f.close()
        db.close()
    (temp / 'seen.sqlite').unlink()
    manifest = {
        'format_version': 2, 'dtype': dtype, 'vocab_size': len(tokenizer),
        'eos_token_id': tokenizer.eos_token_id, 'source': cfg,
        'documents_scanned': seen, 'exact_duplicates_skipped': duplicates,
        'source_mix': source_counts,
        'packing': 'EOS between documents; cross-document attention allowed; final document may be truncated',
        'splits': {s: {'stored_tokens': counts[s], 'target_tokens': counts[s] - 1,
                       'documents_used': docs[s], 'sha256': hashes[s].hexdigest()} for s in goals},
    }
    (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.replace(temp, out)
    print(f'Ready: {out}. Train labels={train_tokens:,}; validation labels={val_tokens:,}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/adamw_ru_500m.json')
    p.add_argument('--output', default='data/russian_mix_10b')
    p.add_argument('--train-tokens', type=int)
    p.add_argument('--val-tokens', type=int)
    prepare(p.parse_args())
