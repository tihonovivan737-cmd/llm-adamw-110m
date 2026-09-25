import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from llm.data import TokenFile, document_key, validation_document, learning_rate
from llm.model import LanguageModel, ModelConfig, make_optimizer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def fixture(directory):
    config = json.loads((ROOT / 'configs/adamw_110m.json').read_text())
    config['model'].update(vocab_size=128, dim=32, layers=2, heads=4, ffn_dim=80, seq_len=16)
    config['training'].update(max_tokens=259, global_batch_tokens=128, micro_batch_size=2,
                              precision='fp32', eval_every=2, save_every=2,
                              eval_tokens=65, log_every=1)
    data = directory / 'data'
    data.mkdir()
    (data / 'tokenizer').mkdir()
    manifest = {'format_version': 1, 'dtype': '<u2', 'vocab_size': 128, 'splits': {}}
    for split, count in [('train', 260), ('val', 66)]:
        arr = np.random.default_rng(count).integers(0, 128, count, dtype=np.uint16).astype('<u2')
        raw = arr.tobytes()
        (data / f'{split}.bin').write_bytes(raw)
        manifest['splits'][split] = {'stored_tokens': count, 'target_tokens': count - 1,
                                    'sha256': hashlib.sha256(raw).hexdigest()}
    (data / 'manifest.json').write_text(json.dumps(manifest))
    cfg_path = directory / 'config.json'
    cfg_path.write_text(json.dumps(config))
    return config, cfg_path, data


class ModelTests(unittest.TestCase):
    def test_production_parameter_count(self):
        cfg = ModelConfig()
        with torch.device('meta'):
            model = LanguageModel(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 109529856)
        self.assertEqual(cfg.parameter_count, 109529856)

    def test_causal_attention_and_gradients(self):
        cfg = ModelConfig(vocab_size=128, dim=32, layers=2, heads=4, ffn_dim=80, seq_len=16)
        model = LanguageModel(cfg).eval()
        x = torch.randint(0, 128, (2, 16))
        changed = x.clone()
        changed[:, 8:] = torch.randint(0, 128, (2, 8))
        torch.testing.assert_close(model(x)[:, :8], model(changed)[:, :8], atol=0, rtol=0)
        model(x, x).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_masked_empty_rank_has_zero_gradient(self):
        model = LanguageModel(ModelConfig(vocab_size=32, dim=16, layers=1, heads=2, ffn_dim=32, seq_len=4))
        x = torch.zeros((1, 4), dtype=torch.long)
        loss = model(x, torch.full_like(x, -100))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in model.parameters()))

    def test_accumulation_equals_full_batch(self):
        torch.manual_seed(42)
        cfg = ModelConfig(vocab_size=32, dim=16, layers=1, heads=2, ffn_dim=32, seq_len=4)
        a = LanguageModel(cfg)
        b = copy.deepcopy(a)
        x = torch.randint(0, 32, (4, 4))
        y = torch.randint(0, 32, (4, 4))
        y[-1, 1:] = -100
        valid = (y != -100).sum()
        (a(x, y) / valid).backward()
        for i in range(4):
            (b(x[i:i+1], y[i:i+1]) / valid).backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)

    def test_token_packing_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = fixture(Path(tmp))
            data = TokenFile(path, 'train')
            labels = []
            for start in range(0, data.target_tokens, 32):
                count = min(32, data.target_tokens - start)
                x, y = data.batch(start, count, 2, 16, torch.device('cpu'))
                labels.extend(y[y != -100].tolist())
                self.assertEqual(int((y != -100).sum()), count)
            self.assertEqual(labels, data.tokens[1:].tolist())
            del data  # Release the memory-mapped file before Windows removes the fixture.

    def test_split_determinism(self):
        key = document_key(' sample document ')
        self.assertEqual(key, document_key('sample document'))
        self.assertEqual(validation_document(key, .02, 42), validation_document(key, .02, 42))

    def test_schedule_endpoints(self):
        cfg = {'max_tokens': 1000, 'warmup_fraction': .1, 'learning_rate': .001, 'min_lr_ratio': .1}
        self.assertEqual(learning_rate(100, cfg), .001)
        self.assertAlmostEqual(learning_rate(1000, cfg), .0001)

    def test_optimizer_groups_no_double_count(self):
        cfg = ModelConfig(vocab_size=32, dim=16, layers=1, heads=2, ffn_dim=32, seq_len=4)
        model = LanguageModel(cfg)
        tc = json.loads((ROOT / 'configs/adamw_110m.json').read_text())['training']
        opt = make_optimizer(model, tc, torch.device('cpu'))
        params = [p for group in opt.param_groups for p in group['params']]
        self.assertEqual(len(params), len({id(p) for p in params}))
        group = next(g for g in opt.param_groups if any(p is model.embedding.weight for p in g['params']))
        self.assertEqual(group['weight_decay'], 0)


class IntegrationTests(unittest.TestCase):
    def test_resume_equals_uninterrupted_and_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config, data = fixture(root)
            base = [sys.executable, str(ROOT / 'train.py'), '--device', 'cpu', '--config', str(config), '--data', str(data)]
            full, resumed = root / 'full', root / 'resumed'
            def call(extra):
                result = subprocess.run(base + extra, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            call(['--output', str(full)])
            call(['--output', str(resumed), '--stop-after-steps', '1'])
            call(['--output', str(resumed), '--resume', str(resumed / 'latest.pt')])
            a = torch.load(full / 'latest.pt', weights_only=False)
            b = torch.load(resumed / 'latest.pt', weights_only=False)
            self.assertEqual(a['tokens_seen'], 259)
            self.assertEqual(b['tokens_seen'], 259)
            self.assertEqual(a['step'], 3)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
            output = call(['--eval-only', '--full-validation', '--resume', str(full / 'latest.pt')])
            metrics = json.loads(output.strip().splitlines()[-1])
            self.assertEqual(metrics['val_tokens'], 65)
            self.assertTrue(np.isfinite(metrics['val_loss']))


if __name__ == '__main__':
    unittest.main()
