from __future__ import annotations

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper


class _IdentityProcessor:
    def unapply_actions(self, actions: np.ndarray) -> np.ndarray:
        return actions


class _RandomFramework:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.request_kwargs: list[dict] = []

    def predict_action(self, *, examples, **kwargs):
        self.request_kwargs.append(kwargs)
        return {
            "normalized_actions": torch.rand(1, 2, 3),
            "geometry": {
                "depth_current": torch.ones(1, 1, 2, 2),
                "depth_future": torch.full((1, 1, 2, 2), 2.0),
                "uvd": torch.zeros(1, 3, 3),
                "uvd_time": torch.tensor([[0.0, 0.0, 0.0]]),
                "uvd_landmark_ids": torch.tensor([[0, 1, 2]]),
            },
        }


def _make_wrapper(framework: _RandomFramework) -> PolicyServerWrapper:
    wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
    wrapper._framework = framework
    wrapper._default_unnorm_key = "libero"
    wrapper._available_unnorm_keys = ["libero"]
    wrapper._get_processor = lambda key: _IdentityProcessor()
    return wrapper


def test_policy_wrapper_scopes_seed_and_forwards_geometry_as_numpy() -> None:
    """Removing scoped RNG or geometry conversion must break this response contract."""
    framework = _RandomFramework()
    wrapper = _make_wrapper(framework)
    example = {"image": [], "lang": "move"}

    torch.manual_seed(1234)
    expected_after_request = torch.rand(4)
    torch.manual_seed(1234)
    first = wrapper.predict_action([example], inference_seed=7)
    observed_after_request = torch.rand(4)
    second = wrapper.predict_action([example], inference_seed=7)
    third = wrapper.predict_action([example], inference_seed=8)
    unseeded = wrapper.predict_action([example])

    np.testing.assert_array_equal(first["actions"], second["actions"])
    assert not np.array_equal(first["actions"], third["actions"])
    torch.testing.assert_close(observed_after_request, expected_after_request)
    assert framework.request_kwargs == [{}, {}, {}, {}]
    assert set(first["geometry"]) == {
        "depth_current",
        "depth_future",
        "uvd",
        "uvd_time",
        "uvd_landmark_ids",
    }
    assert all(isinstance(value, np.ndarray) for value in first["geometry"].values())
    assert first["geometry"]["uvd_landmark_ids"].tolist() == [[0, 1, 2]]
    assert unseeded["actions"].shape == (1, 2, 3)
