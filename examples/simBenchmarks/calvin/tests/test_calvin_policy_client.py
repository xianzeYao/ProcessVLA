import unittest
from unittest.mock import patch
from types import SimpleNamespace

from examples.simBenchmarks.calvin.eval_files import eval_calvin
from deployment.model_server.tools.websocket_policy_client import (
    _connect_with_header_compat,
)


class CurrentModelClient:
    """Network-free stand-in with the current ModelClient constructor."""

    def __init__(
        self,
        unnorm_key=None,
        policy_setup="franka",
        horizon=0,
        action_ensemble=True,
        action_ensemble_horizon=3,
        use_ddim=True,
        num_ddim_steps=10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
        image_size=(224, 224),
        action_stride=None,
    ):
        self.unnorm_key = unnorm_key
        self.host = host
        self.port = port
        self.image_size = tuple(image_size)
        self.action_chunk_size = 8
        self.action_stride = action_stride or 8


class CalvinPolicyClientTest(unittest.TestCase):
    def test_constructs_current_model_client_without_obsolete_checkpoint_keyword(self):
        with patch.object(eval_calvin, "ModelClient", CurrentModelClient):
            try:
                client = eval_calvin.CalvinPolicyClient(
                    host="127.0.0.1",
                    port=5694,
                    resize_size=224,
                    action_stride=8,
                    pretrained_path="/tmp/checkpoint.pt",
                    unnorm_key="franka",
                )
            except TypeError as exc:
                self.fail(str(exc))

        self.assertEqual(client.action_stride, 8)
        self.assertEqual(client.action_chunk_size, 8)
        self.assertEqual(client.client.unnorm_key, "franka")


class WebsocketCompatibilityTest(unittest.TestCase):
    def test_retries_without_ping_options_for_legacy_sync_connector(self):
        def legacy_connect(
            uri,
            *,
            additional_headers=None,
            compression=None,
            max_size=None,
            open_timeout=None,
        ):
            return uri, additional_headers

        try:
            connection = _connect_with_header_compat(
                legacy_connect,
                "ws://127.0.0.1:5794",
                headers={"Authorization": "Api-Key test"},
                compression=None,
                max_size=None,
                open_timeout=150,
                ping_interval=None,
                ping_timeout=60,
            )
        except TypeError as exc:
            self.fail(str(exc))

        self.assertEqual(connection[0], "ws://127.0.0.1:5794")


class CalvinEnvironmentTest(unittest.TestCase):
    def test_cpu_backend_disables_egl_from_validation_config(self):
        env_config = SimpleNamespace(cameras={}, use_egl=True)
        cfg = SimpleNamespace(env=env_config)
        expected_env = object()

        with patch.dict(
            eval_calvin.os.environ,
            {"CALVIN_RENDER_BACKEND": "cpu"},
        ):
            with patch(
                "omegaconf.OmegaConf.load",
                return_value=cfg,
            ), patch.object(
                eval_calvin.hydra.utils,
                "instantiate",
                return_value=expected_env,
            ):
                env = eval_calvin.make_env("/tmp/calvin")

        self.assertIs(env, expected_env)
        self.assertFalse(env_config.use_egl)


if __name__ == "__main__":
    unittest.main()
