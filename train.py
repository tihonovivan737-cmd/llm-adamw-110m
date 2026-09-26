"""Train with AdamW on one device or torchrun DDP, with exact token accounting."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.data import TokenFile, fingerprint, learning_rate
from llm.model import LanguageModel, ModelConfig, make_optimizer


def autocast(device, precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and precision == 'bf16' else nullcontext()


@torch.no_grad()
def evaluate(model, data, token_limit, batch_size, device, precision, rank=0, world=1):
    was_training = model.training
    model.eval()
    limit = min(token_limit, data.target_tokens)
    cap = batch_size * model.cfg.seq_len
    sums = torch.zeros(2, dtype=torch.float64, device=device)
    for offset in range(rank * cap, limit, world * cap):
        count = min(cap, limit - offset)
        x, y = data.batch(offset, count, batch_size, model.cfg.seq_len, device)
        with autocast(device, precision):
            loss = model(x, y)
        sums[0] += loss.double()
        sums[1] += count
    if world > 1:
        dist.all_reduce(sums)
    model.train(was_training)
    mean = (sums[0] / sums[1]).item()
    return {'val_loss': mean, 'perplexity': math.exp(min(mean, 80)), 'val_tokens': int(sums[1].item())}


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def save_checkpoint(path, model, optimizer, config, manifest_hash, tokens, step,
                    best, device, rank, world, best_now=False):
    states = [None] * world
    state = rng_state(device)
    if world > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if rank == 0:
        checkpoint = {'format_version': 1, 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'config': config,
                      'data_fingerprint': manifest_hash, 'tokens_seen': tokens,
                      'step': step, 'best_val_loss': best, 'world_size': world,
                      'rng_states': states, 'torch_version': str(torch.__version__)}
        temporary = path.with_suffix('.tmp')
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        if best_now:
            best_tmp = path.parent / 'best.tmp'
            shutil.copyfile(path, best_tmp)
            os.replace(best_tmp, path.parent / 'best.pt')
    if world > 1:
        dist.barrier()


def check_hashes(directory, manifest):
    for name, info in manifest['splits'].items():
        h = hashlib.sha256()
        with (Path(directory) / f'{name}.bin').open('rb') as f:
            while block := f.read(8 * 1024 * 1024):
                h.update(block)
        if h.hexdigest() != info['sha256']:
            raise ValueError(f'{name}.bin SHA256 mismatch')


def check_continuation(checkpoint, previous_manifest, data_dir, manifest, config):
    """Allow a longer token budget only when the new corpus extends the old one exactly."""
    if checkpoint['data_fingerprint'] != fingerprint(previous_manifest):
        raise ValueError('Saved run manifest does not match the checkpoint')

    old_config = checkpoint['config']
    old_source = dict(previous_manifest['source'])
    new_source = dict(manifest['source'])
    if old_source != old_config['data'] or new_source != config['data']:
        raise ValueError('Dataset manifests do not match their saved configurations')
    old_data = dict(old_config['data'])
    new_data = dict(config['data'])
    old_budget = old_data.pop('train_tokens')
    new_budget = new_data.pop('train_tokens')
    if old_data != new_data or new_budget < old_budget:
        raise ValueError('Continuation requires the same data source and an equal-or-larger train_tokens budget')
    if (new_budget == old_budget
            and checkpoint['data_fingerprint'] != fingerprint(manifest)):
        raise ValueError('Same-budget continuation requires the exact same prepared corpus')

    old_training = dict(old_config['training'])
    new_training = dict(config['training'])
    old_training.pop('max_tokens')
    new_training.pop('max_tokens')
    # A continuation uses a lower, constant learning rate for the second stage.
    for training in (old_training, new_training):
        training.pop('learning_rate')
        training.pop('min_lr_ratio')
    if old_config['model'] != config['model'] or old_training != new_training:
        raise ValueError('Continuation cannot change the model or other training settings')
    if (config['training']['learning_rate'] >= old_config['training']['learning_rate']
            or config['training']['min_lr_ratio'] != 1.0):
        raise ValueError('Continuation requires a lower constant learning rate')
    if config['training']['max_tokens'] <= checkpoint['tokens_seen']:
        raise ValueError('Continuation max_tokens must exceed tokens_seen in the checkpoint')
    if config['training']['max_tokens'] > TokenFile(data_dir, 'train').target_tokens:
        raise ValueError('Continuation max_tokens exceeds the extended train split')

    old_val = previous_manifest['splits']['val']
    new_val = manifest['splits']['val']
    if old_val['target_tokens'] != new_val['target_tokens'] or old_val['sha256'] != new_val['sha256']:
        raise ValueError('Continuation must keep the same validation tokens')

    old_train = previous_manifest['splits']['train']
    prefix_bytes = old_train['stored_tokens'] * np.dtype(previous_manifest['dtype']).itemsize
    h = hashlib.sha256()
    with (Path(data_dir) / 'train.bin').open('rb') as f:
        remaining = prefix_bytes
        while remaining:
            block = f.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError('Extended train split is shorter than the original corpus')
            h.update(block)
            remaining -= len(block)
    if h.hexdigest() != old_train['sha256']:
        raise ValueError('Extended train split is not an exact prefix of the original corpus')


def run(args):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Install a CUDA PyTorch build or pass --device cpu for a small test.')
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.set_num_threads(args.cpu_threads)
    if world > 1:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.seed is not None:
        config['training']['seed'] = args.seed
    tc = config['training']
    mc = ModelConfig(**config['model'])
    if tc['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Supported precision: fp32 or bf16')
    if tc['precision'] == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16; use an fp32 test config')
    for key in ['micro_batch_size', 'global_batch_tokens', 'max_tokens', 'eval_every',
                'eval_tokens', 'save_every', 'log_every']:
        if tc[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 < tc['warmup_fraction'] < 1 or not 0 <= tc['min_lr_ratio'] <= 1:
        raise ValueError('Invalid LR schedule')
    micro_cap = tc['micro_batch_size'] * mc.seq_len
    if tc['global_batch_tokens'] % (world * micro_cap):
        raise ValueError('global_batch_tokens must be divisible by world_size * micro_batch_size * seq_len')
    random.seed(tc['seed'])
    np.random.seed(tc['seed'])
    torch.manual_seed(tc['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(tc['seed'])
    train_data, val_data = TokenFile(args.data, 'train'), TokenFile(args.data, 'val')
    manifest = train_data.manifest
    if manifest['vocab_size'] != mc.vocab_size or tc['max_tokens'] > train_data.target_tokens:
        raise ValueError('Vocabulary mismatch or requested training exceeds prepared token budget')
    check_hashes(args.data, manifest)
    data_hash = fingerprint(manifest)
    model = LanguageModel(mc).to(device)
    optimizer = make_optimizer(model, tc, device)
    tokens, step, best = 0, 0, float('inf')
    checkpoint = None
    if args.resume:
        # Only load checkpoints you trust: optimizer/RNG states require pickle.
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if args.continue_training:
            previous_manifest_path = Path(args.output) / 'data_manifest.json'
            if not previous_manifest_path.exists():
                raise ValueError('--continue-training requires the previous run data_manifest.json')
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding='utf-8'))
            check_continuation(checkpoint, previous_manifest, args.data, manifest, config)
        elif checkpoint['data_fingerprint'] != data_hash or checkpoint['config'] != config:
            raise ValueError('Resume requires exactly the same dataset manifest and configuration')
        if not args.eval_only and checkpoint['world_size'] != world:
            raise ValueError('Resume training with the same world size; evaluation may use one GPU')
        model.load_state_dict(checkpoint['model'])
        tokens, step, best = checkpoint['tokens_seen'], checkpoint['step'], checkpoint['best_val_loss']
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
    elif args.eval_only:
        raise ValueError('--eval-only requires --resume checkpoint.pt')
    network = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None) if world > 1 else model
    if checkpoint and not args.eval_only:
        restore_rng(checkpoint['rng_states'][rank], device)
    checkpoint = None
    out = Path(args.output)
    if not args.eval_only:
        if rank == 0:
            if out.exists() and any(out.iterdir()) and not args.resume:
                raise FileExistsError('Run directory is not empty; use --resume or a different --output')
            out.mkdir(parents=True, exist_ok=True)
            (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
            (out / 'data_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            if not (out / 'tokenizer').exists():
                shutil.copytree(Path(args.data) / 'tokenizer', out / 'tokenizer')
        if world > 1:
            dist.barrier()

    def log(record):
        if rank == 0:
            print(json.dumps(record), flush=True)
            if not args.eval_only:
                with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + '\n')

    log({'event': 'start', 'parameters': model.parameter_count, 'world_size': world,
         'device': str(device), 'precision': tc['precision'] if device.type == 'cuda' else 'fp32',
         'tokens_seen': tokens, 'target_tokens': tc['max_tokens'],
         'gradient_accumulation_steps': tc['global_batch_tokens'] // (world * micro_cap)})
    if args.eval_only:
        log(evaluate(model, val_data, val_data.target_tokens if args.full_validation else tc['eval_tokens'],
                     tc['micro_batch_size'], device, tc['precision'], rank, world))
        return
    if tokens == 0:
        initial = evaluate(model, val_data, tc['eval_tokens'], tc['micro_batch_size'], device, tc['precision'], rank, world)
        log({'event': 'validation', 'step': 0, 'tokens_seen': 0, **initial})
        # best.pt always corresponds to a saved, trained checkpoint.
    model.train()
    while tokens < tc['max_tokens'] and (args.stop_after_steps is None or step < args.stop_after_steps):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        step_tokens = min(tc['global_batch_tokens'], tc['max_tokens'] - tokens)
        micro_steps = math.ceil(step_tokens / (world * micro_cap))
        lr = learning_rate(tokens + step_tokens, tc)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device, dtype=torch.float64)
        for micro in range(micro_steps):
            relative = (micro * world + rank) * micro_cap
            count = max(0, min(micro_cap, step_tokens - relative))
            x, y = train_data.batch(tokens + relative, count, tc['micro_batch_size'], mc.seq_len, device)
            sync = network.no_sync() if world > 1 and micro < micro_steps - 1 else nullcontext()
            with sync:
                with autocast(device, tc['precision']):
                    loss_sum = network(x, y)
                    # DDP averages across ranks; undo that before global token normalization.
                    loss = loss_sum * (world / step_tokens)
                loss.backward()
            total_loss += loss_sum.detach().double()
        if world > 1:
            dist.all_reduce(total_loss)
        if not torch.isfinite(total_loss):
            raise FloatingPointError('Non-finite loss; restart from latest.pt after investigating')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        tokens += step_tokens
        step += 1
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        final = tokens == tc['max_tokens'] or (args.stop_after_steps is not None and step >= args.stop_after_steps)
        if step == 1 or step % tc['log_every'] == 0 or final:
            log({'event': 'train', 'step': step, 'tokens_seen': tokens, 'step_tokens': step_tokens,
                 'train_loss': total_loss.item() / step_tokens, 'lr': lr,
                 'grad_norm_before_clip': float(grad_norm), 'step_seconds': elapsed,
                 'tokens_per_second': step_tokens / elapsed,
                 'peak_vram_gb': torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0})
        improved = False
        if step % tc['eval_every'] == 0 or final:
            result = evaluate(model, val_data, tc['eval_tokens'], tc['micro_batch_size'], device, tc['precision'], rank, world)
            improved = result['val_loss'] < best
            best = min(best, result['val_loss'])
            log({'event': 'validation', 'step': step, 'tokens_seen': tokens, **result})
        if step % tc['save_every'] == 0 or final or improved:
            save_checkpoint(out / 'latest.pt', model, optimizer, config, data_hash, tokens,
                            step, best, device, rank, world, improved)
    log({'event': 'finished' if tokens == tc['max_tokens'] else 'paused', 'tokens_seen': tokens, 'step': step})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/adamw_ru_500m.json')
    p.add_argument('--data', default='data/russian_mix_10b')
    p.add_argument('--output', default='runs/qwen35_ru_500m_seed42')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int)
    p.add_argument('--resume', type=Path)
    p.add_argument('--continue-training', action='store_true',
                   help='Continue into a larger, verified train split with a second-stage config')
    p.add_argument('--stop-after-steps', type=int, help='Absolute step at which to checkpoint and exit; schedule is unchanged')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--full-validation', action='store_true', help='With --eval-only: use all prepared validation tokens')
    return p


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
