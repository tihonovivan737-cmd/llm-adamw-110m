"""Render metrics.jsonl as Markdown training and per-GPU memory tables."""
import argparse
import json
from pathlib import Path
import time


TRAIN_COLUMNS = (
    ('step', 'Step'),
    ('tokens_seen', 'Tokens seen'),
    ('step_tokens', 'Step tokens'),
    ('train_loss', 'Train loss'),
    ('lr', 'LR'),
    ('grad_norm_before_clip', 'Grad norm'),
    ('step_seconds', 'Seconds'),
    ('tokens_per_second', 'Tokens/s'),
    ('peak_vram_gb', 'Peak VRAM (GB)'),
)
MEMORY_COLUMNS = (
    ('step', 'Step'),
    ('gpu', 'GPU'),
    ('device_used_gb', 'Device used / total (GB)'),
    ('process_allocated_gb', 'Allocated / reserved (GB)'),
    ('model_parameters_gb', 'Weights (GB)'),
    ('gradients_gb', 'Gradients (GB)'),
    ('adamw_states_gb', 'AdamW states (GB)'),
    ('other_current_allocated_gb', 'Other current (GB)'),
    ('other_at_step_peak_approx_gb', 'Other at peak, approx (GB)'),
)


def cell(value, digits=3):
    if value is None:
        return '—'
    if isinstance(value, (int, float)):
        if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
            return f'{int(value):,}'
        return f'{value:.{digits}f}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def write_header(file, title, columns):
    file.write(f'# {title}\n\n')
    file.write('| ' + ' | '.join(label for _, label in columns) + ' |\n')
    file.write('| ' + ' | '.join('---' for _ in columns) + ' |\n')
    file.flush()


def append_record(record, train_file, memory_file):
    if record.get('event') != 'train':
        return False
    train_file.write('| ' + ' | '.join(cell(record.get(key)) for key, _ in TRAIN_COLUMNS) + ' |\n')
    for gpu in record.get('gpu_memory', []):
        values = dict(record)
        values.update(gpu)
        device_usage = f"{cell(gpu.get('device_used_gb'))} / {cell(gpu.get('device_total_gb'))}"
        process_usage = f"{cell(gpu.get('process_allocated_gb'))} / {cell(gpu.get('process_reserved_gb'))}"
        values['device_used_gb'] = device_usage
        values['process_allocated_gb'] = process_usage
        memory_file.write('| ' + ' | '.join(cell(values.get(key)) for key, _ in MEMORY_COLUMNS) + ' |\n')
    train_file.flush()
    memory_file.flush()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logfile', type=Path, help='Run metrics.jsonl')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory (defaults to the run directory)')
    parser.add_argument('--follow', action='store_true',
                        help='Keep watching the log and append new training steps')
    args = parser.parse_args()
    log_path = args.logfile
    if not log_path.is_file():
        parser.error(f'log file not found: {log_path}')
    output_dir = args.output_dir or log_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / 'training_steps.md'
    memory_path = output_dir / 'gpu_memory.md'

    with train_path.open('w', encoding='utf-8') as train_file, \
            memory_path.open('w', encoding='utf-8') as memory_file:
        write_header(train_file, 'Training steps', TRAIN_COLUMNS)
        write_header(memory_file, 'GPU memory by training step', MEMORY_COLUMNS)
        with log_path.open('r', encoding='utf-8') as log_file:
            for line in log_file:
                try:
                    append_record(json.loads(line), train_file, memory_file)
                except json.JSONDecodeError:
                    continue
            print(f'Таблицы обновлены: {train_path} и {memory_path}', flush=True)
            if args.follow:
                while True:
                    position = log_file.tell()
                    line = log_file.readline()
                    if not line or not line.endswith('\n'):
                        log_file.seek(position)
                        time.sleep(0.5)
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if append_record(record, train_file, memory_file):
                        print(f"Таблицы обновлены до шага {record.get('step')}", flush=True)


if __name__ == '__main__':
    main()
