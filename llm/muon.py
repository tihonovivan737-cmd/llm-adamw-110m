"""Muon matrix optimizer and an AdamW auxiliary optimizer for other weights."""

import torch
from torch.optim import Optimizer


@torch.no_grad()
def orthogonalize_update(gradient, steps):
    """Approximate the matrix zeroth power with quintic Newton-Schulz steps."""
    if gradient.ndim != 2:
        raise ValueError('Muon updates require 2D matrices')
    update = gradient.to(torch.bfloat16)
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.mT
    update = update / (update.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = update @ update.mT
        polynomial = b * gram + c * (gram @ gram)
        update = a * update + polynomial @ update
    if transposed:
        update = update.mT
    return update


class Muon(Optimizer):
    """Per-rank Muon optimizer; DDP has already synchronized its gradients."""

    def __init__(self, params, lr, weight_decay, momentum=0.95, ns_steps=5, nesterov=True):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay,
                                      momentum=momentum, ns_steps=ns_steps,
                                      nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(parameter)
                momentum = state['momentum_buffer']
                momentum.lerp_(gradient, 1.0 - group['momentum'])
                update = gradient.lerp(momentum, group['momentum']) if group['nesterov'] else momentum
                update = orthogonalize_update(update, group['ns_steps'])
                update *= max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
                if group['weight_decay']:
                    parameter.mul_(1.0 - group['lr'] * group['weight_decay'])
                parameter.add_(update.to(parameter.dtype), alpha=-group['lr'])
        return loss


class MuonAdamW:
    """One checkpointable optimizer facade over local Muon and AdamW states."""

    def __init__(self, muon_parameters, adamw_groups, config, device):
        self.muon = Muon(
            muon_parameters,
            lr=config['muon_learning_rate'],
            weight_decay=config['muon_weight_decay'],
            momentum=config['muon_momentum'],
            ns_steps=config['muon_ns_steps'],
            nesterov=config['muon_nesterov'],
        )
        self.adamw = torch.optim.AdamW(
            adamw_groups,
            lr=config['learning_rate'],
            betas=tuple(config['betas']),
            eps=config['eps'],
            fused=device.type == 'cuda',
        )
        for group in self.muon.param_groups:
            group['optimizer_kind'] = 'muon'
        for group in self.adamw.param_groups:
            group['optimizer_kind'] = 'adamw'

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    @property
    def state(self):
        return {**self.muon.state, **self.adamw.state}

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {'muon': self.muon.state_dict(), 'adamw': self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict['muon'])
        self.adamw.load_state_dict(state_dict['adamw'])
