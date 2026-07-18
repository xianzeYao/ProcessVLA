import torch
from torch import nn

from starVLA.training.cot_test_diagnostics import (
    collect_module_grad_norms,
    install_module_grad_norm_hooks,
    set_module_grad_norm_collection,
)


class _ToyCotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry_query = nn.Linear(2, 2, bias=False)
        self.depth_decoder = nn.Linear(2, 2, bias=False)
        self.uvd_head = nn.Linear(2, 2, bias=False)
        self.action_model = nn.Linear(2, 2, bias=False)


class _Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


def test_gradient_hooks_are_removed_after_one_collection():
    model = _ToyCotModel()
    wrapper = _Wrapper(model)
    hook_state = install_module_grad_norm_hooks(wrapper)
    assert hook_state["handles"]

    sum(parameter.sum() for parameter in wrapper.parameters()).backward()
    metrics = collect_module_grad_norms(wrapper, hook_state=hook_state)

    assert metrics["grad/query_norm"] > 0.0
    assert metrics["grad/query_local_rms"] > 0.0
    assert hook_state["handles"] == []
    assert all(value is None for value in hook_state["sums"].values())
