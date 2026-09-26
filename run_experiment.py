"""Prepare the corpus if requested, then launch single-GPU or DDP training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/adamw_ru_500m.json')
    p.add_argument('--data', default='data/russian_mix_10b')
    p.add_argument('--output', default='runs/qwen35_ru_500m_seed42')
    p.add_argument('--gpus', type=int, default=2)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume output/latest.pt')
    p.add_argument('--continue-training', action='store_true',
                   help='Continue from latest.pt using a larger, prefix-verified corpus')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stop-after-steps', type=int)
    args = p.parse_args()
    if args.gpus < 1:
        p.error('--gpus must be positive')
    if args.continue_training and not args.resume:
        p.error('--continue-training requires --resume')
    if os.name == 'nt' and args.gpus > 1:
        p.error('NCCL DDP requires Linux/WSL2. Native Windows supports --gpus 1 here.')
    import torch
    if torch.cuda.device_count() < args.gpus:
        p.error(f'Requested {args.gpus} GPUs, found {torch.cuda.device_count()} CUDA devices')
    root = Path(__file__).resolve().parent
    if args.prepare and not (Path(args.data) / 'manifest.json').exists():
        subprocess.run([sys.executable, str(root / 'prepare_data.py'), '--config', args.config,
                        '--output', args.data], check=True)
    if not (Path(args.data) / 'manifest.json').exists():
        p.error('Prepared data missing. Run prepare_data.py or add --prepare.')
    command = [sys.executable]
    if args.gpus > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={args.gpus}']
    command += [str(root / 'train.py'), '--config', args.config, '--data', args.data,
                '--output', args.output, '--seed', str(args.seed)]
    if args.resume:
        checkpoint = Path(args.output) / 'latest.pt'
        if not checkpoint.exists():
            p.error(f'Checkpoint not found: {checkpoint}')
        command += ['--resume', str(checkpoint)]
    if args.continue_training:
        command += ['--continue-training']
    if args.stop_after_steps is not None:
        command += ['--stop-after-steps', str(args.stop_after_steps)]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
