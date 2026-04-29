from __future__ import annotations

import re
from functools import partial

import torch
import torch.nn as nn
from torch import optim


def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) == 2
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 5:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, nesterov: bool = True) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class MuSGD(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, use_muon=use_muon)
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_SGD"] = torch.zeros_like(p)
                    update = muon_update(grad, state["momentum_buffer"], beta=group["momentum"], nesterov=group["nesterov"])
                    p.add_(update.reshape(p.shape), alpha=-(lr * self.muon))
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state["momentum_buffer_SGD"].mul_(group["momentum"]).add_(grad)
                    sgd_update = grad.add(state["momentum_buffer_SGD"], alpha=group["momentum"]) if group["nesterov"] else state["momentum_buffer_SGD"]
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
            else:
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = grad.add(state["momentum_buffer"], alpha=group["momentum"]) if group["nesterov"] else state["momentum_buffer"]
                    p.add_(update, alpha=-lr)
        return loss


def build_optimizer(
    model: nn.Module,
    name: str = "MuSGD",
    lr: float = 0.001,
    momentum: float = 0.9,
    decay: float = 1e-5,
    iterations: float = 1e5,
):
    g = [{}, {}, {}, {}]
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    if name == "auto":
        name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", 0.001, 0.9)
    use_muon = name == "MuSGD"

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if param.ndim >= 2 and use_muon:
                g[3][fullname] = param
            elif "bias" in fullname:
                g[2][fullname] = param
            elif isinstance(module, bn) or "logit_scale" in fullname:
                g[1][fullname] = param
            else:
                g[0][fullname] = param
    if not use_muon:
        g = [x.values() for x in g[:3]]

    optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
    name = {x.lower(): x for x in optimizers}.get(name.lower())
    if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
        optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optim_args = dict(lr=lr, momentum=momentum)
    elif name in {"SGD", "MuSGD"}:
        optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer '{name}' not found in {optimizers}.")

    g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
    g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
    g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
    muon, sgd = (0.2, 1.0)
    if use_muon:
        g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
        pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg")
        g_ = []
        for x in g:
            p = x.pop("params")
            p1 = [v for k, v in p.items() if pattern.search(k)]
            p2 = [v for k, v in p.items() if not pattern.search(k)]
            g_.extend([{"params": p1, **x, "lr": lr * 3}, {"params": p2, **x}])
        g = g_
    return getattr(optim, name, partial(MuSGD, muon=muon, sgd=sgd))(params=g)
