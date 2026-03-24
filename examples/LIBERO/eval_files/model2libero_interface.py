from collections import deque  # 环形队列，用于图像历史
from typing import Optional, Sequence
import os
import cv2 as cv  # 图像缩放
import matplotlib.pyplot as plt  # 可视化
import numpy as np

# WebSocket 客户端，连到策略服务器
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler  # 自适应动作集成
from typing import Dict
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from starVLA.model.tools import read_mode_config  # 读取模型配置与归一化统计
from starVLA.model.framework.vlac_utils import (
    DatasetSeededVLACReferenceResolver,
    OnlineVLACSignalState,
    VLACOnlineSubprocessClient,
)


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=True,
        action_ensemble_horizon: Optional[int] = 3,  # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
        signal_infer_source: Optional[str] = None,
        vlac_reference_video_path: Optional[str] = None,
        vlac_reference_mode: Optional[str] = None,
        vlac_reference_dataset_name: Optional[str] = None,
        vlac_reference_seed: Optional[int] = None,
        vlac_reference_data_root_dir: Optional[str] = None,
        vlac_reference_data_mix: Optional[str] = None,
        vlac_signal_kind: Optional[str] = None,
        vlac_skip: Optional[int] = None,
        vlac_frame_skip: Optional[bool] = None,
        vlac_ref_num: Optional[int] = None,
        vlac_batch_num: Optional[int] = None,
        vlac_rich: Optional[bool] = None,
        vlac_think: Optional[bool] = None,
        vlac_device: Optional[str] = None,
        vlac_python: Optional[str] = None,
        vlac_model_path: Optional[str] = None,
        vlac_model_type: Optional[str] = None,
        vlac_repo_root: Optional[str] = None,
    ) -> None:

        # build client to connect server policy
        # 建立到策略服务器的 WebSocket 客户端
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon  # 0  # 图像历史长度（未使用时为 0）
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        # 粘滞动作/抓手重复的状态
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.task_id = None
        self.episode_idx = None
        self.image_history = deque(maxlen=self.horizon)  # 图像历史队列
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(
            policy_ckpt_path=policy_ckpt_path)

        parts = Path(policy_ckpt_path).parts
        save_hidden_dir = (
            f"{parts[-3]}_{parts[-1]}" if len(parts) >= 3 else "_".join(parts)
        )
        self.save_hidden_dir = save_hidden_dir
        default_signal_dump_root = (
            Path.cwd() / "results" / "libero_signal_dumps"
        ).resolve()
        self.signal_dump_root = Path(
            os.environ.get("PROCESSVLA_SIGNAL_DUMP_ROOT", str(default_signal_dump_root))
        ).expanduser().resolve()
        model_config, _ = read_mode_config(policy_ckpt_path)
        signal_cfg = dict(model_config.get("signal", {}) or {})
        vla_cfg = dict(model_config.get("datasets", {}).get("vla_data", {}) or {})
        vlac_cfg = dict(signal_cfg.get("vlac_online", {}) or {})
        default_signal_infer_source = self.normalize_infer_source(
            signal_cfg.get("infer_source", "none")
        )
        self.signal_infer_source = self.resolve_infer_source(
            override=signal_infer_source,
            default=default_signal_infer_source,
        )
        print(
            "*** "
            f"signal_infer_source: {self.signal_infer_source} "
            "***"
        )
        self.vlac_reference_mode = self.resolve_vlac_reference_mode(
            override=vlac_reference_mode,
            default=vlac_cfg.get("reference_mode", None),
            explicit_video_path=vlac_reference_video_path or vlac_cfg.get("reference_video_path", None),
            data_root_dir=vlac_reference_data_root_dir or vlac_cfg.get("reference_data_root_dir", vla_cfg.get("data_root_dir", None)),
            data_mix=vlac_reference_data_mix or vlac_cfg.get("reference_data_mix", vla_cfg.get("data_mix", None)),
        )
        self.vlac_reference_video_path = self.resolve_optional_str(
            vlac_reference_video_path,
            vlac_cfg.get("reference_video_path", None),
        )
        self.vlac_reference_dataset_name = self.resolve_optional_str(
            vlac_reference_dataset_name,
            vlac_cfg.get("reference_dataset_name", None),
        )
        self.vlac_reference_seed = int(
            self.resolve_optional_int(vlac_reference_seed, vlac_cfg.get("reference_seed", 42))
        )
        self.vlac_reference_data_root_dir = self.resolve_required_str(
            vlac_reference_data_root_dir,
            vlac_cfg.get("reference_data_root_dir", vla_cfg.get("data_root_dir", None)),
            field_name="vlac reference data_root_dir",
        )
        self.vlac_reference_data_mix = self.resolve_required_str(
            vlac_reference_data_mix,
            vlac_cfg.get("reference_data_mix", vla_cfg.get("data_mix", None)),
            field_name="vlac reference data_mix",
        )
        self.vlac_reference_video_backend = str(
            vlac_cfg.get("reference_video_backend", vla_cfg.get("video_backend", "torchvision_av"))
        )
        self.vlac_signal_kind = self.resolve_required_str(
            vlac_signal_kind,
            vlac_cfg.get("signal_kind", "value"),
            field_name="vlac signal_kind",
        ).lower()
        self.vlac_skip = int(self.resolve_optional_int(vlac_skip, vlac_cfg.get("skip", 5)))
        self.vlac_frame_skip = bool(
            self.resolve_optional_bool(vlac_frame_skip, vlac_cfg.get("frame_skip", False))
        )
        self.vlac_ref_num = int(self.resolve_optional_int(vlac_ref_num, vlac_cfg.get("ref_num", 6)))
        self.vlac_batch_num = int(
            self.resolve_optional_int(vlac_batch_num, vlac_cfg.get("batch_num", 5))
        )
        self.vlac_rich = bool(self.resolve_optional_bool(vlac_rich, vlac_cfg.get("rich", False)))
        self.vlac_think = bool(
            self.resolve_optional_bool(vlac_think, vlac_cfg.get("think", False))
        )
        self.vlac_device = str(vlac_device) if vlac_device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.vlac_python = self.resolve_optional_str(
            vlac_python,
            vlac_cfg.get("python", None),
        )
        self.vlac_model_path = self.resolve_optional_str(
            vlac_model_path,
            vlac_cfg.get("model_path", None),
        )
        self.vlac_model_type = self.resolve_required_str(
            vlac_model_type,
            vlac_cfg.get("model_type", "internvl2"),
            field_name="vlac model_type",
        )
        self.vlac_repo_root = self.resolve_optional_str(
            vlac_repo_root,
            vlac_cfg.get("repo_root", None),
        )
        self.vlac_online_state = None
        self.vlac_reference_resolver = None
        self.vlac_subprocess_client = None
        if self.signal_infer_source == "vlac_online":
            if self.vlac_python is not None or self.vlac_model_path is not None or self.vlac_repo_root is not None:
                if not all([self.vlac_python, self.vlac_model_path, self.vlac_repo_root]):
                    raise ValueError(
                        "vlac_online subprocess mode requires vlac_python, vlac_model_path, and vlac_repo_root together."
                    )
                self.vlac_subprocess_client = VLACOnlineSubprocessClient(
                    vlac_python=self.vlac_python,
                    model_path=self.vlac_model_path,
                    model_type=self.vlac_model_type,
                    repo_root=self.vlac_repo_root,
                    device=self.vlac_device,
                )
            if self.vlac_reference_mode == "explicit_video":
                if not self.vlac_reference_video_path:
                    raise ValueError(
                        "signal.infer_source='vlac_online' with reference_mode='explicit_video' requires vlac_reference_video_path."
                    )
            elif self.vlac_reference_mode == "dataset_same_task_seeded":
                self.vlac_reference_resolver = DatasetSeededVLACReferenceResolver(
                    data_root_dir=self.vlac_reference_data_root_dir,
                    data_mix=self.vlac_reference_data_mix,
                    dataset_name=self.vlac_reference_dataset_name,
                    reference_seed=self.vlac_reference_seed,
                    video_backend=self.vlac_reference_video_backend,
                )
            else:
                raise ValueError(
                    f"Unsupported vlac reference_mode={self.vlac_reference_mode!r}. "
                    "Supported values are: explicit_video, dataset_same_task_seeded."
                )

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str, task_id: int | None = None, episode_idx: int | None = None) -> None:
        self.task_description = task_description
        self.task_id = task_id
        self.episode_idx = episode_idx
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        # 重置粘滞/抓手状态
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        if self.signal_infer_source == "vlac_online":
            selected_reference_video_path = self.resolve_vlac_reference_video_path(
                task_description=task_description,
                task_id=task_id,
                episode_idx=episode_idx,
            )
            if (
                self.vlac_online_state is not None
                and self.vlac_online_state.reference_video_path == selected_reference_video_path
            ):
                self.vlac_online_state.reset(task_description)
            else:
                self.vlac_online_state = OnlineVLACSignalState(
                    instruction=task_description,
                    reference_video_path=selected_reference_video_path,
                    signal_kind=self.vlac_signal_kind,
                    skip=self.vlac_skip,
                    frame_skip=self.vlac_frame_skip,
                    ref_num=self.vlac_ref_num,
                    batch_num=self.vlac_batch_num,
                    rich=self.vlac_rich,
                    think=self.vlac_think,
                    device=self.vlac_device,
                    subprocess_client=self.vlac_subprocess_client,
                )
            print(
                "*** "
                f"vlac_online reference_mode: {self.vlac_reference_mode}, "
                f"reference_video_path: {selected_reference_video_path} "
                "***"
            )

    def step(
        self,
        example: dict,
        step: int = 0,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        执行一步推理：
        :param image: 输入图像 (H, W, 3)，uint8
        :param task_description: 任务描述文本
        :return: (原始动作，处理后动作)
        """

        task_description = example.get("lang", None)
        images = example["image"]  # list of images for history

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]  # 统一图像尺寸
        example["image"] = images
        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        if self.signal_infer_source == "vlac_online":
            self.vlac_online_state.append_frame(images[0])

        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            task_slug = (self.task_description or "unknown_task").replace(
                " ", "_").replace("/", "_")
            episode_tag = self.episode_idx if self.episode_idx is not None else 0
            hiddenstates_file_path = (
                self.signal_dump_root
                / self.save_hidden_dir
                / "hidden_states"
                / f"{task_slug}_episode_{episode_tag}"
                / f"step_{step}_last_hidden_states_meta.pt"
            )
            os.makedirs(hiddenstates_file_path.parent, exist_ok=True)
            print(f"Hidden states will be saved to: {hiddenstates_file_path}")
            # Attach save path into payload so websocket can carry it to server.
            vla_input["hidden_save_path"] = hiddenstates_file_path
            if self.signal_infer_source == "vlac_online":
                vla_input["signal_override"] = float(
                    self.vlac_online_state.compute_current_signal()
                )
            vla_input["signal_infer_source"] = self.signal_infer_source
            response = self.client.predict_action(
                vla_input)  # 每个 chunk 入口调用一次模型
            try:
                # B, chunk, D
                normalized_actions = response["data"]["normalized_actions"]
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(
                    f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]
            # 反归一化到物理控制量
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)

        # 取当前步在 chunk 内的那一帧动作
        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            # range [0, 1]; 1 = open; 0 = close
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }

        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(
            action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(
            action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(
            normalized_actions[:, 6] < 0.5, 0, 1)  # 抓手阈值化
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) *
            (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        读取动作归一化统计（兼容旧接口）。
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(
            policy_ckpt_path)  # read config and norm_stats

        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(
            policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config['framework']['action_model']['future_action_window_size'] + 1

    @staticmethod
    def normalize_infer_source(value) -> str:
        if value is None:
            return "none"
        return str(value).strip().lower()

    @staticmethod
    def resolve_optional_str(override, default) -> Optional[str]:
        value = default if override is None else override
        if value is None:
            return None
        text = str(value).strip()
        return None if not text else text

    @staticmethod
    def resolve_required_str(override, default, *, field_name: str) -> str:
        value = ModelClient.resolve_optional_str(override, default)
        if value is None:
            raise ValueError(f"Missing required {field_name}.")
        return value

    @staticmethod
    def resolve_optional_int(override, default) -> int:
        value = default if override is None else override
        return int(value)

    @staticmethod
    def resolve_optional_bool(override, default) -> bool:
        value = default if override is None else override
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    @staticmethod
    def normalize_vlac_reference_mode(value) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        return None if not text else text

    @staticmethod
    def resolve_vlac_reference_mode(
        *,
        override: Optional[str],
        default,
        explicit_video_path,
        data_root_dir,
        data_mix,
    ) -> str:
        mode = ModelClient.normalize_vlac_reference_mode(
            override if override is not None else default
        )
        if mode is None:
            if explicit_video_path:
                mode = "explicit_video"
            elif data_root_dir and data_mix:
                mode = "dataset_same_task_seeded"
            else:
                raise ValueError(
                    "Unable to infer vlac reference_mode. Provide vlac_reference_mode explicitly or configure "
                    "either vlac_reference_video_path or reference dataset settings."
                )
        if mode not in {"explicit_video", "dataset_same_task_seeded"}:
            raise ValueError(
                f"Unsupported vlac reference_mode={mode!r}. "
                "Supported values are: explicit_video, dataset_same_task_seeded."
            )
        return mode

    def resolve_vlac_reference_video_path(
        self,
        *,
        task_description: str,
        task_id: int | None,
        episode_idx: int | None,
    ) -> str:
        if self.vlac_reference_mode == "explicit_video":
            if not self.vlac_reference_video_path:
                raise ValueError("Missing vlac_reference_video_path for reference_mode='explicit_video'.")
            return str(Path(self.vlac_reference_video_path).expanduser().resolve())
        if self.vlac_reference_mode == "dataset_same_task_seeded":
            if self.vlac_reference_resolver is None:
                raise ValueError("vlac reference resolver was not initialized.")
            return self.vlac_reference_resolver.select_reference_video_path(
                instruction=task_description,
                task_id=task_id,
                episode_idx=episode_idx,
                dataset_name=self.vlac_reference_dataset_name,
            )
        raise ValueError(
            f"Unsupported vlac reference_mode={self.vlac_reference_mode!r}."
        )

    @staticmethod
    def resolve_infer_source(override: Optional[str], default: str) -> str:
        source = default if override is None else override
        source = ModelClient.normalize_infer_source(source)
        if source not in {"none", "liv_online", "vlac_online"}:
            raise ValueError(
                f"Unsupported signal.infer_source={source!r}. Supported values are: none, liv_online, vlac_online."
            )
        return source

    @staticmethod
    def get_signal_infer_source(policy_ckpt_path) -> str:
        model_config, _ = read_mode_config(
            policy_ckpt_path)  # read config and norm_stats
        signal_cfg = model_config.get("signal", {})
        return ModelClient.normalize_infer_source(
            signal_cfg.get("infer_source", "none")
        )

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size),
                          interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(
                pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        辅助函数（兼容旧版），校验/补充 unnorm_key。
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
