from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn.functional as F


def build_hidden_state_save_payload(
    qwen_inputs,
    last_hidden: torch.Tensor,
    base_last_hidden: torch.Tensor,
    batch_images,
    instructions: List[str],
    tokenizer,
    image_token_id: int,
) -> Dict[str, Any]:
    input_ids = qwen_inputs["input_ids"].detach().cpu()
    attention_mask = qwen_inputs["attention_mask"].detach().cpu()
    image_grid_thw = qwen_inputs.get("image_grid_thw")
    image_grid_thw = image_grid_thw.detach().cpu(
    ) if image_grid_thw is not None else None
    image_mask = input_ids == int(image_token_id)
    pad_token_id = tokenizer.pad_token_id

    special_token_ids = {
        "image_token_id": int(image_token_id),
        "pad_token_id": int(pad_token_id) if pad_token_id is not None else None,
        "vision_start_id": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "vision_end_id": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        "im_start_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }
    decoded_inputs = tokenizer.batch_decode(
        input_ids, skip_special_tokens=False)
    token_counts = {
        "sequence_length": attention_mask.sum(dim=1),
        "num_image_tokens": image_mask.sum(dim=1),
    }
    if image_grid_thw is not None:
        token_counts["image_grid_thw"] = image_grid_thw

    return {
        "hidden_states_last": last_hidden.detach().cpu(),
        "hidden_states_last_base": base_last_hidden.detach().cpu(),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_mask": image_mask,
        "special_token_ids": special_token_ids,
        "token_counts": token_counts,
        "decoded_inputs": decoded_inputs,
        "images": batch_images,
        "instructions": instructions,
    }


@lru_cache(maxsize=1)
def load_liv_runtime():
    import clip
    import torchvision.transforms as trans
    from liv import load_liv

    liv_model = load_liv()
    if isinstance(liv_model, torch.nn.DataParallel):
        liv_model = liv_model.module
    liv_model = liv_model.eval()
    transform = trans.Compose([trans.ToTensor()])
    return liv_model, clip, transform


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_repo_import_path(repo_name: str, src_subdir: Optional[str] = None) -> None:
    repo_root = _workspace_root() / repo_name
    if not repo_root.exists():
        raise ImportError(
            f"Local repo '{repo_name}' not found under {_workspace_root()}.")
    import_path = repo_root / src_subdir if src_subdir is not None else repo_root
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _ensure_optional_module_stubs() -> None:
    class _NoOpLogger:
        def __getattr__(self, _name: str):
            def _noop(*args, **kwargs):
                return None
            return _noop

    def _module_missing(module_name: str) -> bool:
        if module_name in sys.modules:
            return False
        try:
            return importlib.util.find_spec(module_name) is None
        except ValueError:
            return False

    has_loguru = not _module_missing("loguru")
    has_codetiming = not _module_missing("codetiming")

    if not has_loguru:
        loguru_stub = types.ModuleType("loguru")
        loguru_stub.logger = _NoOpLogger()
        sys.modules["loguru"] = loguru_stub

    if "robometer.utils.logger" not in sys.modules and not has_loguru:
        robometer_logger_stub = types.ModuleType("robometer.utils.logger")
        _logger = _NoOpLogger()
        _noop = lambda *args, **kwargs: None
        robometer_logger_stub.loguru_logger = _logger
        robometer_logger_stub.get_logger = lambda *args, **kwargs: _logger
        robometer_logger_stub.rank_0_info = _noop
        robometer_logger_stub.rank_0_warning = _noop
        robometer_logger_stub.rank_0_debug = _noop
        robometer_logger_stub.rank_0_trace = _noop
        robometer_logger_stub.rank_0_debug2 = _noop
        robometer_logger_stub.trace = _noop
        robometer_logger_stub.debug2 = _noop
        sys.modules["robometer.utils.logger"] = robometer_logger_stub

    if not has_codetiming:
        codetiming_stub = types.ModuleType("codetiming")

        class _Timer:
            def __init__(self, name: str = "", logger: Any = None, *args, **kwargs):
                self.name = name
                self.logger = logger
                self.last = 0.0
                self._start = 0.0

            def __enter__(self):
                self._start = time.time()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.last = time.time() - self._start
                return False

        codetiming_stub.Timer = _Timer
        sys.modules["codetiming"] = codetiming_stub


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load module {module_name!r} from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_camera_list(sample_images: Any) -> List[Any]:
    if isinstance(sample_images, (list, tuple)):
        return list(sample_images)
    return [sample_images]


def _as_time_window(window_or_sample: Any) -> List[Any]:
    if isinstance(window_or_sample, (list, tuple)):
        if len(window_or_sample) == 0:
            return []
        first_item = window_or_sample[0]
        if isinstance(first_item, (list, tuple)):
            return list(window_or_sample)
    return [window_or_sample]


def _coerce_image_chw(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        try:
            from PIL import Image

            if isinstance(image, Image.Image):
                image = np.asarray(image)
        except Exception:
            pass
        tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected image rank 3, got shape={tuple(tensor.shape)}.")
    if tensor.shape[0] in {1, 3}:
        chw = tensor
    elif tensor.shape[-1] in {1, 3}:
        chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError(
            "Expected image in CHW or HWC format with 1 or 3 channels, "
            f"got shape={tuple(tensor.shape)}."
        )
    if chw.shape[0] == 1:
        chw = chw.expand(3, -1, -1)
    return chw.contiguous()


def _to_unit_float_chw(image: Any) -> torch.Tensor:
    chw = _coerce_image_chw(image).to(dtype=torch.float32)
    if bool(torch.max(chw) > 1.0) or bool(torch.min(chw) < 0.0):
        chw = (chw / 255.0).clamp(0.0, 1.0)
    return chw


def _resize_chw_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if image.shape[-2:] == size:
        return image
    resized = F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _to_hwc_uint8_image(image: Any) -> np.ndarray:
    chw = _coerce_image_chw(image)
    if chw.dtype.is_floating_point:
        if bool(torch.max(chw) <= 1.0) and bool(torch.min(chw) >= 0.0):
            chw = chw * 255.0
        chw = chw.clamp(0.0, 255.0)
    else:
        chw = chw.to(dtype=torch.float32).clamp(0.0, 255.0)
    return chw.permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()


DEFAULT_ROBOMETER_MODEL_PATH = os.environ.get(
    "CRITIC4VLA_ROBOMETER_MODEL_PATH",
    "ROBOMETER_model_path",
)
DEFAULT_ROBOMETER_QWEN_SNAPSHOT_PATH = os.environ.get(
    "CRITIC4VLA_ROBOMETER_QWEN_SNAPSHOT_PATH",
    "ROBOMETER_qwen_snapshot_path",
)
DEFAULT_ROBOMETER_UNSLOTH_SNAPSHOT_PATH = os.environ.get(
    "CRITIC4VLA_ROBOMETER_UNSLOTH_SNAPSHOT_PATH",
    "ROBOMETER_unsloth_snapshot_path",
)
DEFAULT_ROBODOPAMINE_MODEL_PATH = os.environ.get(
    "CRITIC4VLA_ROBODOPAMINE_MODEL_PATH",
    "ROBODOPAMINE_model_path",
)
DEFAULT_VLAC_MODEL_PATH = os.environ.get(
    "CRITIC4VLA_VLAC_MODEL_PATH",
    "VLAC_model_path",
)
DEFAULT_VLAC_MODEL_TYPE = os.environ.get(
    "CRITIC4VLA_VLAC_MODEL_TYPE", "internvl2")
ROBOMETER_MAX_IMAGE_SIDE = 480
ROBOMETER_MAX_IMAGE_PIXELS = 1024 * 1024


def _robometer_find_checkpoint_dir(model_path: str | Path) -> Path:
    resolved = Path(model_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Robometer checkpoint not found: {resolved}")
    return resolved


def _robometer_find_config_path(checkpoint_dir: Path) -> Path:
    candidate_paths = [
        checkpoint_dir / "config.yaml",
        checkpoint_dir.parent / "config.yaml",
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find Robometer config.yaml in {checkpoint_dir} or its parent."
    )


def _robometer_resolve_config_value(value: Any, config_dict: Dict[str, Any]) -> Any:
    if not (isinstance(value, str) and value.startswith("${") and value.endswith("}")):
        return value

    path = value[2:-1].split(".")
    current: Any = config_dict
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return value
        current = current[key]
    if current == value:
        return value
    return _robometer_resolve_config_value(current, config_dict)


def _robometer_load_config(checkpoint_dir: Path):
    import yaml

    raw_config = yaml.safe_load(
        _robometer_find_config_path(checkpoint_dir).read_text())
    if not isinstance(raw_config, dict):
        raise ValueError(
            "Robometer config.yaml did not parse to a dictionary.")

    model_section = raw_config.get("model", {}) or {}
    data_section = raw_config.get("data", {}) or {}
    loss_section = raw_config.get("loss", {}) or {}

    def _get(section: Dict[str, Any], key: str, default: Any) -> Any:
        return _robometer_resolve_config_value(section.get(key, default), raw_config)

    model_cfg = types.SimpleNamespace(
        base_model_id=str(_get(model_section, "base_model_id",
                          "Qwen/Qwen3-VL-4B-Instruct")),
        model_type=str(_get(model_section, "model_type", "default")),
        torch_dtype=str(_get(model_section, "torch_dtype", "bfloat16")),
        trust_remote_code=bool(_get(model_section, "trust_remote_code", True)),
        average_temporal_patches=bool(
            _get(model_section, "average_temporal_patches", False)),
        use_per_frame_progress_token=bool(
            _get(model_section, "use_per_frame_progress_token", False)),
        frame_pooling=str(_get(model_section, "frame_pooling", "mean")),
        frame_pooling_attn_temperature=float(
            _get(model_section, "frame_pooling_attn_temperature", 1.0)),
        use_multi_image=bool(_get(model_section, "use_multi_image", False)),
        use_peft=bool(_get(model_section, "use_peft", False)),
        progress_loss_type=str(_get(model_section, "progress_loss_type", _get(
            loss_section, "progress_loss_type", "l2"))),
        progress_discrete_bins=int(_get(model_section, "progress_discrete_bins", _get(
            loss_section, "progress_discrete_bins", 10)) or 10),
    )

    data_cfg = types.SimpleNamespace(
        resized_height=_get(data_section, "resized_height", None),
        resized_width=_get(data_section, "resized_width", None),
        use_multi_image=bool(
            _get(data_section, "use_multi_image", model_cfg.use_multi_image)),
        use_per_frame_progress_token=bool(
            _get(data_section, "use_per_frame_progress_token",
                 model_cfg.use_per_frame_progress_token)
        ),
        max_frames=int(_get(data_section, "max_frames", 16) or 16),
    )

    loss_cfg = types.SimpleNamespace(
        progress_loss_type=str(
            _get(loss_section, "progress_loss_type", model_cfg.progress_loss_type)),
        progress_discrete_bins=int(_get(
            loss_section, "progress_discrete_bins", model_cfg.progress_discrete_bins) or 10),
    )

    model_cfg.use_multi_image = bool(
        model_cfg.use_multi_image or data_cfg.use_multi_image)
    model_cfg.use_per_frame_progress_token = bool(
        model_cfg.use_per_frame_progress_token or data_cfg.use_per_frame_progress_token
    )

    return types.SimpleNamespace(model=model_cfg, data=data_cfg, loss=loss_cfg)


def _robometer_processor_model_id(base_model_id: str) -> str:
    if base_model_id.startswith("unsloth/") and "Qwen" in base_model_id:
        return "Qwen/" + base_model_id.split("/", 1)[1]
    return base_model_id


def _robometer_snapshot_has_model_weights(snapshot_path: Path) -> bool:
    if not snapshot_path.exists() or not snapshot_path.is_dir():
        return False
    direct_candidates = [
        snapshot_path / "model.safetensors",
        snapshot_path / "pytorch_model.bin",
        snapshot_path / "model.safetensors.index.json",
        snapshot_path / "pytorch_model.bin.index.json",
    ]
    if any(path.is_file() for path in direct_candidates):
        return True
    return any(snapshot_path.glob("model-*.safetensors"))


def _robometer_snapshot_has_processor_files(snapshot_path: Path) -> bool:
    if not snapshot_path.exists() or not snapshot_path.is_dir():
        return False
    processor_candidates = [
        snapshot_path / "preprocessor_config.json",
        snapshot_path / "processor_config.json",
        snapshot_path / "tokenizer_config.json",
    ]
    return any(path.is_file() for path in processor_candidates)


def _robometer_resolve_base_model_path(base_model_id: str, *, for_processor: bool) -> str:
    qwen_snapshot = Path(DEFAULT_ROBOMETER_QWEN_SNAPSHOT_PATH).expanduser()
    unsloth_snapshot = Path(
        DEFAULT_ROBOMETER_UNSLOTH_SNAPSHOT_PATH).expanduser()

    if "Qwen" not in base_model_id:
        return base_model_id

    if base_model_id.startswith("unsloth/"):
        if for_processor and _robometer_snapshot_has_processor_files(qwen_snapshot):
            return str(qwen_snapshot.resolve())
        if _robometer_snapshot_has_model_weights(unsloth_snapshot):
            return str(unsloth_snapshot.resolve())
        if _robometer_snapshot_has_model_weights(qwen_snapshot):
            return str(qwen_snapshot.resolve())
        if qwen_snapshot.exists():
            return str(qwen_snapshot.resolve())
        if unsloth_snapshot.exists():
            return str(unsloth_snapshot.resolve())
        return base_model_id

    if for_processor and _robometer_snapshot_has_processor_files(qwen_snapshot):
        return str(qwen_snapshot.resolve())
    if _robometer_snapshot_has_model_weights(qwen_snapshot):
        return str(qwen_snapshot.resolve())
    if _robometer_snapshot_has_model_weights(unsloth_snapshot):
        return str(unsloth_snapshot.resolve())
    if qwen_snapshot.exists():
        return str(qwen_snapshot.resolve())
    if unsloth_snapshot.exists():
        return str(unsloth_snapshot.resolve())
    return base_model_id


def _robometer_is_valid_processor(processor: Any) -> bool:
    return (
        hasattr(processor, "tokenizer")
        and hasattr(processor, "__call__")
        and callable(getattr(processor, "apply_chat_template", None))
    )


def _robometer_load_processor(
    primary_path: str,
    *,
    fallback_paths: Sequence[str],
    trust_remote_code: bool,
) -> Any:
    from transformers import AutoProcessor

    candidate_paths: List[str] = []
    for path in [primary_path, *fallback_paths]:
        if path and path not in candidate_paths:
            candidate_paths.append(path)

    errors: List[str] = []
    explicit_processor_classes: List[Any] = []
    try:
        from transformers import Qwen3VLProcessor

        explicit_processor_classes.append(Qwen3VLProcessor)
    except Exception:
        pass
    try:
        from transformers import Qwen2_5_VLProcessor

        explicit_processor_classes.append(Qwen2_5_VLProcessor)
    except Exception:
        pass

    for model_path in candidate_paths:
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                do_sample_frames=False,
                padding_side="right",
                local_files_only=True,
            )
            if _robometer_is_valid_processor(processor):
                if processor.tokenizer.pad_token is None:
                    processor.tokenizer.pad_token = processor.tokenizer.eos_token
                return processor
            errors.append(
                f"AutoProcessor({model_path}) returned {type(processor).__name__}, not a multimodal processor."
            )
        except Exception as exc:
            errors.append(f"AutoProcessor({model_path}) failed: {exc}")

        for processor_cls in explicit_processor_classes:
            try:
                processor = processor_cls.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
                if _robometer_is_valid_processor(processor):
                    if processor.tokenizer.pad_token is None:
                        processor.tokenizer.pad_token = processor.tokenizer.eos_token
                    return processor
                errors.append(
                    f"{processor_cls.__name__}({model_path}) returned {type(processor).__name__}, not a multimodal processor."
                )
            except Exception as exc:
                errors.append(
                    f"{processor_cls.__name__}({model_path}) failed: {exc}")

    raise RuntimeError(
        "Unable to construct a valid Robometer processor from local snapshots. "
        + " | ".join(errors)
    )


def _robometer_resize_pil(image, max_side: int = ROBOMETER_MAX_IMAGE_SIDE, max_pixels: int = ROBOMETER_MAX_IMAGE_PIXELS):
    from PIL import Image

    pil = image.convert("RGB")
    width, height = pil.size
    scale_side = min(1.0, max_side / float(max(width, height)))
    scale_area = (max_pixels / float(width * height)
                  ) ** 0.5 if (width * height) > max_pixels else 1.0
    scale = min(scale_side, scale_area)
    if scale < 1.0:
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        pil = pil.resize((new_width, new_height), resample=Image.BICUBIC)
    return pil


def _robometer_add_special_tokens_and_resize(processor: Any, base_model: Any) -> None:
    special_tokens = [
        "<|split_token|>",
        "<|reward_token|>",
        "<|pref_token|>",
        "<|sim_token|>",
        "<|prog_token|>",
    ]
    num_added = 0
    vocab = processor.tokenizer.get_vocab()
    for token in special_tokens:
        if token not in vocab:
            num_added += processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": [token]}
            )
    if num_added > 0:
        base_model.resize_token_embeddings(len(processor.tokenizer))


def _robometer_prepare_frames_for_conversation(
    video_frames: Sequence[np.ndarray],
    exp_config: Any,
):
    from PIL import Image

    pil_frames = [Image.fromarray(_to_hwc_uint8_image(frame))
                  for frame in video_frames]
    use_multi_image = bool(exp_config.model.use_multi_image)
    base_model_id = str(exp_config.model.base_model_id)

    if use_multi_image:
        return pil_frames, {}

    if "Qwen" in base_model_id or "Molmo" in base_model_id:
        if exp_config.data.resized_height is not None and exp_config.data.resized_width is not None:
            return pil_frames, {
                "resized_height": int(exp_config.data.resized_height),
                "resized_width": int(exp_config.data.resized_width),
            }
        resized_frames = [_robometer_resize_pil(frame) for frame in pil_frames]
        return resized_frames, {}

    raise ValueError(
        f"Robometer minimal inference currently supports Qwen/Molmo checkpoints, got {base_model_id!r}."
    )


def _robometer_add_vision_content_to_list(
    content_list: List[Dict[str, Any]],
    frames_or_video: Any,
    content_extras: Dict[str, Any],
    *,
    use_multi_image: bool,
    use_per_frame_progress_token: bool,
) -> None:
    if use_multi_image:
        for image in frames_or_video:
            content_list.append(
                {
                    "type": "image",
                    "image": image,
                    **content_extras,
                }
            )
            if use_per_frame_progress_token:
                content_list.append({"type": "text", "text": "<|prog_token|>"})
        return

    content_list.append(
        {
            "type": "video",
            "video": frames_or_video,
            "sample_fps": 1.0,
            **content_extras,
        }
    )


def _robometer_process_conversations(
    conversations: List[List[Dict[str, Any]]],
    processor: Any,
    *,
    base_model_id: str,
    max_length: int = 1024,
) -> Dict[str, torch.Tensor]:
    from qwen_vl_utils import process_vision_info

    if "Qwen" not in base_model_id and "Molmo" not in base_model_id:
        raise ValueError(
            f"Robometer minimal inference only supports Qwen/Molmo checkpoints, got {base_model_id!r}."
        )

    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=True,
            enable_thinking=False,
            fps=1,
        )
        for message in conversations
    ]

    is_qwen3 = "Qwen3" in base_model_id or "Molmo2" in base_model_id
    process_kwargs: Dict[str, Any] = {
        "return_video_kwargs": True,
        "return_video_metadata": is_qwen3,
    }
    if (
        is_qwen3
        and hasattr(processor, "image_processor")
        and hasattr(processor.image_processor, "patch_size")
    ):
        process_kwargs["image_patch_size"] = processor.image_processor.patch_size

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        conversations, **process_kwargs
    )

    if is_qwen3 and video_inputs is not None and len(video_inputs) > 0:
        if isinstance(video_inputs[0], tuple) and len(video_inputs[0]) == 2:
            videos, video_metadatas = zip(*video_inputs)
            videos = list(videos)
            video_metadatas = list(video_metadatas)
        else:
            videos = video_inputs
            video_metadatas = None
    else:
        videos = video_inputs if video_inputs else None
        video_metadatas = None

    processor_kwargs: Dict[str, Any] = {
        "text": texts,
        "images": image_inputs,
        "padding": True,
        "truncation": False,
        "max_length": int(max_length),
        "return_tensors": "pt",
        "do_resize": False,
    }
    if videos is not None:
        processor_kwargs["videos"] = videos
    if is_qwen3:
        if video_metadatas is not None:
            processor_kwargs["video_metadata"] = video_metadatas
        if video_kwargs:
            processor_kwargs.update(video_kwargs)

    return processor(**processor_kwargs)


def _robometer_build_progress_inputs(
    video_frames: Sequence[np.ndarray],
    instruction: str,
    processor: Any,
    exp_config: Any,
) -> Dict[str, Any]:
    frames_or_video, content_extras = _robometer_prepare_frames_for_conversation(
        video_frames, exp_config
    )
    prompt = (
        f"The task for the robot is '{instruction}'. Given the trajectory video, "
        "predict the task progress at each frame, how far along the robot is "
        "towards completing the task, a float between 0 and 1, where 0 is the "
        "starting state and 1 is when the task is completed. If the robot is "
        "not performing the same task, predict 0 progress."
    )
    content_list: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    _robometer_add_vision_content_to_list(
        content_list,
        frames_or_video,
        content_extras,
        use_multi_image=bool(exp_config.model.use_multi_image),
        use_per_frame_progress_token=bool(
            exp_config.model.use_per_frame_progress_token),
    )
    conversation = [[{"role": "user", "content": content_list}]]
    batch_inputs = _robometer_process_conversations(
        conversation,
        processor,
        base_model_id=str(exp_config.model.base_model_id),
    )
    batch_inputs["task"] = [str(instruction)]
    return batch_inputs


def _robometer_load_state_dict_from_checkpoint(
    model: Any,
    checkpoint_dir: Path,
    *,
    ignored_prefixes: Sequence[str] = (),
) -> None:
    from safetensors.torch import load_file

    safetensor_paths = sorted(checkpoint_dir.glob("*.safetensors"))
    safetensor_paths = [
        path for path in safetensor_paths
        if not any(path.name.startswith(prefix) for prefix in ignored_prefixes)
    ]

    if not safetensor_paths:
        pytorch_bin = checkpoint_dir / "pytorch_model.bin"
        if pytorch_bin.is_file():
            state_dict = torch.load(pytorch_bin, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No Robometer checkpoint weights found in {checkpoint_dir}."
            )
    else:
        state_dict: Dict[str, torch.Tensor] = {}
        for path in safetensor_paths:
            state_dict.update(load_file(str(path)))

    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("similarity_head.")
    }

    model_keys = set(model.state_dict().keys())
    remapped_state_dict: Dict[str, torch.Tensor] = {}
    for checkpoint_key, checkpoint_value in state_dict.items():
        if checkpoint_key in model_keys:
            remapped_state_dict[checkpoint_key] = checkpoint_value
            continue

        potential_keys: List[str] = []
        if checkpoint_key.startswith("model.model."):
            potential_keys.append(
                checkpoint_key.replace(
                    "model.model.", "model.base_model.model.model.", 1)
            )
            potential_keys.append(
                checkpoint_key.replace("model.model.", "model.", 1)
            )
        if checkpoint_key.startswith("model."):
            potential_keys.append(checkpoint_key.replace("model.", "", 1))
        if not checkpoint_key.startswith("model."):
            potential_keys.append(f"model.{checkpoint_key}")
        if checkpoint_key.startswith("model.") and not checkpoint_key.startswith("model.base_model."):
            _, remainder = checkpoint_key.split(".", 1)
            potential_keys.append(f"model.base_model.{remainder}")

        matched_key = next(
            (key for key in potential_keys if key in model_keys), None)
        remapped_state_dict[matched_key or checkpoint_key] = checkpoint_value

    model.load_state_dict(remapped_state_dict, strict=False)


@lru_cache(maxsize=4)
def _load_robometer_runtime(model_path: str, device_name: str):
    _ensure_repo_import_path("robometer")
    _ensure_optional_module_stubs()
    checkpoint_dir = _robometer_find_checkpoint_dir(model_path)
    exp_config = _robometer_load_config(checkpoint_dir)

    from robometer.models.rbm import RBM
    from transformers import AutoProcessor, Qwen2_5_VLModel

    try:
        from transformers import Qwen3VLModel
    except ImportError:
        Qwen3VLModel = None

    device = torch.device(device_name)
    model_cfg = exp_config.model
    processor_model_id = _robometer_resolve_base_model_path(
        _robometer_processor_model_id(model_cfg.base_model_id),
        for_processor=True,
    )
    base_model_id = str(model_cfg.base_model_id)
    base_model_path = _robometer_resolve_base_model_path(
        base_model_id,
        for_processor=False,
    )
    processor = _robometer_load_processor(
        processor_model_id,
        fallback_paths=[base_model_path],
        trust_remote_code=bool(model_cfg.trust_remote_code),
    )

    torch_dtype = getattr(torch, str(model_cfg.torch_dtype), torch.bfloat16)
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": bool(model_cfg.trust_remote_code),
        "local_files_only": True,
    }

    if "Qwen3" in base_model_id or "qwen3" in base_model_id.lower():
        if Qwen3VLModel is None:
            raise ImportError(
                "Current transformers build does not provide Qwen3VLModel."
            )
        base_model = Qwen3VLModel.from_pretrained(
            base_model_path,
            **model_kwargs,
        )
    elif "Qwen2.5" in base_model_id or "Qwen2" in base_model_id:
        base_model = Qwen2_5_VLModel.from_pretrained(
            base_model_path,
            **model_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported Robometer base model for minimal local inference: {base_model_id!r}."
        )

    _robometer_add_special_tokens_and_resize(processor, base_model)

    has_adapter_files = (
        (checkpoint_dir / "adapter_config.json").is_file()
        and (
            (checkpoint_dir / "adapter_model.safetensors").is_file()
            or (checkpoint_dir / "adapter_model.bin").is_file()
        )
    )

    if has_adapter_files:
        from peft import PeftModel

        base_model = PeftModel.from_pretrained(
            base_model,
            str(checkpoint_dir),
            local_files_only=True,
            is_trainable=False,
        )

    reward_model = RBM(
        config=base_model.config,
        processor=processor,
        tokenizer=processor.tokenizer,
        base_model=base_model,
        base_model_id=base_model_id,
        model_config=model_cfg,
    )

    if has_adapter_files:
        custom_heads_path = checkpoint_dir / "custom_heads.safetensors"
        if custom_heads_path.is_file():
            _robometer_load_state_dict_from_checkpoint(
                reward_model,
                checkpoint_dir,
                ignored_prefixes=("adapter_model",),
            )
    else:
        _robometer_load_state_dict_from_checkpoint(
            reward_model,
            checkpoint_dir,
            ignored_prefixes=("custom_heads", "adapter_model"),
        )

    reward_model = reward_model.to(device)
    reward_model.eval()

    loss_config = exp_config.loss
    is_discrete = str(
        getattr(loss_config, "progress_loss_type", "l2")).lower() == "discrete"
    num_bins = int(
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(model_cfg, "progress_discrete_bins", 10)
    )
    return exp_config, processor.tokenizer, processor, reward_model, bool(is_discrete), num_bins


@lru_cache(maxsize=4)
def _load_vlac_runtime(device_name: str):
    if not DEFAULT_VLAC_MODEL_PATH:
        raise ValueError(
            "VLAC selected but no default model path is configured. "
            "Set CRITIC4VLA_VLAC_MODEL_PATH in the environment or edit "
            "DEFAULT_VLAC_MODEL_PATH in singal_utils.py."
        )

    _ensure_repo_import_path("VLAC")
    from evo_vlac import GAC_model

    critic = GAC_model(tag="critic")
    critic.init_model(
        model_path=DEFAULT_VLAC_MODEL_PATH,
        model_type=DEFAULT_VLAC_MODEL_TYPE,
        device_map=device_name,
    )
    critic.temperature = 0.5
    critic.top_k = 1
    critic.set_config()
    critic.set_system_prompt()
    return critic


@lru_cache(maxsize=2)
def _load_robodopamine_runtime(model_path: str):
    module_path = _workspace_root() / "Robo-Dopamine" / "examples" / "inference.py"
    if not module_path.exists():
        raise ImportError(
            f"Robo-Dopamine inference script not found: {module_path}")
    module = _load_module_from_path(
        "critic4vla_robodopamine_inference", module_path)
    runtime = module.GRMInference(model_path=model_path, max_image_num=8)
    return runtime, module


def _pad_or_trim_curve(curve: np.ndarray, target_length: int) -> np.ndarray:
    if curve.size == 0:
        return np.full(target_length, np.nan, dtype=np.float32)
    if curve.shape[0] < target_length:
        curve = np.pad(curve, (0, target_length - curve.shape[0]), mode="edge")
    elif curve.shape[0] > target_length:
        curve = curve[:target_length]
    return curve.astype(np.float32, copy=False)


def _robometer_forward_model(
    model: Any,
    batch_inputs: Dict[str, Any],
    sample_type: str = "progress",
):
    with torch.no_grad():
        if "rewind" in model.__class__.__name__.lower():
            model_output, extra = model(
                video_embeddings=batch_inputs.get("video_embeddings"),
                text_embeddings=batch_inputs.get("text_embeddings"),
                sample_type=sample_type,
                timing_raw=None,
            )
        else:
            model_output, extra = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                pixel_values=batch_inputs.get("pixel_values", None),
                pixel_values_videos=batch_inputs.get(
                    "pixel_values_videos", None),
                image_grid_thw=batch_inputs.get("image_grid_thw", None),
                video_grid_thw=batch_inputs.get("video_grid_thw", None),
                second_per_grid_ts=batch_inputs.get(
                    "second_per_grid_ts", None),
                sample_type=sample_type,
                timing_raw=None,
            )
    return model_output, extra


def _robometer_compute_batch_outputs(
    model: Any,
    batch_inputs: Dict[str, torch.Tensor],
    sample_type: str,
    is_discrete_mode: bool = False,
) -> Dict[str, Any]:
    _ensure_repo_import_path("robometer")
    from robometer.models.utils import convert_bins_to_continuous

    model.eval()
    model_output, _ = _robometer_forward_model(
        model, batch_inputs, sample_type=sample_type)
    results: Dict[str, Any] = {}

    if sample_type == "preference" and model_output.pref_logits is not None:
        logits = model_output.pref_logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        results.update({
            "predictions": preds.detach().cpu().tolist(),
            "prediction_probs": probs.detach().cpu().tolist(),
            "preference_labels": batch_inputs["preference_labels"].cpu().tolist(),
        })

    progress_logits = model_output.progress_logits
    if progress_logits is not None and isinstance(progress_logits, dict) and sample_type == "progress":
        progress_pred = []
        seq_A = progress_logits.get("A")
        seq_A_list = [seq_A[i]
                      for i in range(seq_A.shape[0])] if seq_A is not None else []
        for seq_A_item in seq_A_list:
            if seq_A_item is None:
                progress_pred.append([])
            elif is_discrete_mode:
                continuous_pred = convert_bins_to_continuous(
                    seq_A_item.detach().cpu().float())
                progress_pred.append(
                    continuous_pred.numpy().flatten().tolist())
            else:
                progress_pred.append(
                    seq_A_item.detach().cpu().flatten().tolist())

        if not progress_pred:
            batch_size = len(batch_inputs.get("task", []))
            progress_pred = [[] for _ in range(batch_size)]
        results["progress_pred"] = progress_pred

        if model_output.success_logits is not None:
            success_pred = model_output.success_logits["A"]
            success_probs = torch.sigmoid(success_pred)
            results["outputs_success"] = {
                "success_probs": success_probs.detach().cpu().tolist(),
            }

    return results


def compute_liv_signal(
    batch_images,
    instructions: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    liv_model, clip_module, transform = load_liv_runtime()
    liv_model = liv_model.to(device)
    batch_size = len(instructions)

    image_tensor = torch.stack(
        [transform(sample_images[0]) for sample_images in batch_images], dim=0
    ).to(device=device, non_blocking=True)
    text_tokens = clip_module.tokenize(
        [str(text) for text in instructions]).to(device=device)

    with torch.no_grad():
        img_embedding = liv_model(input=image_tensor, modality="vision")
        text_embedding = liv_model(input=text_tokens, modality="text")
        signal = liv_model.sim(img_embedding, text_embedding)

    signal = torch.as_tensor(signal, device=device, dtype=dtype).reshape(-1)
    if signal.numel() == 1 and batch_size > 1:
        signal = signal.expand(batch_size)
    elif signal.numel() != batch_size:
        raise ValueError(
            f"LIV signal size mismatch: got {signal.numel()}, expected {batch_size}"
        )
    return signal


def compute_liv_curve_signal(
    primary_frames: Sequence[np.ndarray],
    instruction: str,
    device: torch.device,
    *,
    frame_indices: Optional[Sequence[int]] = None,
    batch_size: int = 5,
) -> np.ndarray:
    if frame_indices is None:
        frame_indices = range(len(primary_frames))

    sampled_values: List[np.ndarray] = []
    frame_indices = [int(idx) for idx in frame_indices]
    for start in range(0, len(frame_indices), int(batch_size)):
        chunk_indices = frame_indices[start:start + int(batch_size)]
        batch_images = [[primary_frames[idx]] for idx in chunk_indices]
        signal = compute_liv_signal(
            batch_images=batch_images,
            instructions=[instruction] * len(chunk_indices),
            device=device,
            dtype=torch.float32,
        )
        sampled_values.append(signal.detach().cpu().numpy().astype(np.float32))

    if not sampled_values:
        return np.array([], dtype=np.float32)
    return np.concatenate(sampled_values, axis=0)


def compute_robometer_curve_signal(
    video_frames: Sequence[np.ndarray],
    instruction: str,
    device: torch.device,
    *,
    signal_kind: str = "progress",
) -> np.ndarray:
    frames_array = np.asarray(video_frames, dtype=np.uint8)
    if frames_array.ndim != 4:
        raise ValueError(
            f"Robometer expects frames with shape [T,H,W,C], got {tuple(frames_array.shape)}."
        )

    exp_config, tokenizer, processor, reward_model, is_discrete, num_bins = _load_robometer_runtime(
        DEFAULT_ROBOMETER_MODEL_PATH, str(device)
    )
    progress_inputs = _robometer_build_progress_inputs(
        frames_array,
        instruction,
        processor,
        exp_config,
    )
    for key, value in progress_inputs.items():
        if isinstance(value, torch.Tensor):
            progress_inputs[key] = value.to(device)

    with torch.no_grad():
        results = _robometer_compute_batch_outputs(
            reward_model,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
        )

    progress_pred = results.get("progress_pred", [])
    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get(
        "success_probs", []) if outputs_success else []

    if signal_kind == "progress":
        curve = np.array(progress_pred[0], dtype=np.float32) if progress_pred else np.array(
            [], dtype=np.float32)
    elif signal_kind == "success":
        curve = np.array(success_probs[0], dtype=np.float32) if success_probs else np.array(
            [], dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported Robometer signal_kind={signal_kind!r}; expected 'progress' or 'success'."
        )

    return _pad_or_trim_curve(curve, len(video_frames))


def compute_robodopamine_curve_signal(
    video_path: str | Path,
    instruction: str,
    *,
    wrist_video_path: Optional[str | Path] = None,
    right_wrist_video_path: Optional[str | Path] = None,
    frame_interval: int = 30,
    batch_size: int = 5,
    goal_image_path: Optional[str | Path] = None,
    eval_mode: str = "incremental",
) -> np.ndarray:
    if eval_mode not in {"incremental", "forward", "backward"}:
        raise ValueError(
            f"Unsupported Robo-Dopamine eval_mode={eval_mode!r}; "
            "expected 'incremental', 'forward', or 'backward'."
        )
    runtime, module = _load_robodopamine_runtime(
        DEFAULT_ROBODOPAMINE_MODEL_PATH)

    video_path = Path(video_path).expanduser().resolve()
    left_wrist_path = (
        Path(wrist_video_path).expanduser().resolve()
        if wrist_video_path is not None
        else video_path
    )
    right_wrist_path = (
        Path(right_wrist_video_path).expanduser().resolve()
        if right_wrist_video_path is not None
        else left_wrist_path
    )
    resolved_goal = (
        None if goal_image_path is None else str(
            Path(goal_image_path).expanduser().resolve())
    )

    total_frames = int(module.get_frame_count(video_path)[1])
    sampled_indices = module.make_sample_indices_by_interval(
        total_frames, int(frame_interval))

    with tempfile.TemporaryDirectory(prefix="critic4vla_robodopamine_") as temp_root:
        run_root = Path(
            runtime.run_pipeline(
                cam_high_path=str(video_path),
                cam_left_path=str(left_wrist_path),
                cam_right_path=str(right_wrist_path),
                out_root=temp_root,
                task=str(instruction),
                frame_interval=int(frame_interval),
                batch_size=int(batch_size),
                goal_image=resolved_goal,
                eval_mode=str(eval_mode),
                visualize=False,
            )
        )
        pred_path = run_root / "pred_vllm.json"
        with open(pred_path, "r", encoding="utf-8") as handle:
            predictions = json.load(handle)

    progress_values = np.asarray(
        [float(item.get("progress", 0.0)) for item in predictions],
        dtype=np.float32,
    )
    if progress_values.size == 0:
        return np.zeros(total_frames, dtype=np.float32)

    step_indices = np.asarray(
        sampled_indices[1:1 + progress_values.shape[0]], dtype=np.int32)
    expanded = np.zeros(total_frames, dtype=np.float32)
    first_step = int(step_indices[0])
    if first_step < total_frames:
        expanded[:first_step] = 0.0
    for idx, step in enumerate(step_indices):
        start = int(step)
        end = int(step_indices[idx + 1]) if idx + \
            1 < len(step_indices) else total_frames
        if start >= total_frames:
            continue
        expanded[start:min(end, total_frames)] = progress_values[idx]
    return expanded


def compute_vlac_curve_signal(
    video_path: str | Path,
    instruction: str,
    fps: float,
    *,
    frame_count: int,
    reference_video_path: Optional[str | Path] = None,
    signal_kind: str = "value",
    ref_num: int = 6,
    batch_num: int = 5,
    skip: int = 5,
    rich: bool = False,
    frame_skip: bool = False,
    think: bool = False,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if device is None:
        raise ValueError("'device' is required for VLAC curve computation.")
    critic = _load_vlac_runtime(str(device))

    video_path = Path(video_path)
    resolved_reference = None if reference_video_path is None else Path(
        reference_video_path)
    need_done = signal_kind == "done"
    _, value_list, critic_list, done_list = critic.web_trajectory_critic(
        task_description=instruction,
        main_video_path=str(video_path),
        reference_video_path=None if resolved_reference is None else str(
            resolved_reference),
        batch_num=int(batch_num),
        ref_num=int(ref_num),
        think=bool(think),
        skip=int(skip),
        rich=bool(rich),
        reverse_eval=False,
        output_path=str(video_path.parent),
        fps=float(fps),
        frame_skip=bool(frame_skip),
        done_flag=need_done,
        in_context_done=resolved_reference is not None,
        done_threshold=0.9,
        video_output=False,
    )

    if signal_kind == "value":
        curve = np.asarray(value_list, dtype=np.float32)
        if curve.size > 0 and float(np.nanmax(curve)) > 1.0:
            curve = curve / 100.0
    elif signal_kind == "critic":
        curve = np.asarray([0.0] + [float(v)
                           for v in critic_list], dtype=np.float32)
    elif signal_kind == "done":
        curve = np.full(frame_count, np.nan, dtype=np.float32) if done_list is None else np.asarray(
            done_list, dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported VLAC signal_kind={signal_kind!r}; expected 'value', 'critic', or 'done'."
        )

    return _pad_or_trim_curve(curve, frame_count)


# Backward-compatible aliases while the call sites migrate to compute_*_curve_signal names.
compute_robometer_signal = compute_robometer_curve_signal
compute_vlac_signal = compute_vlac_curve_signal
compute_robometer_rollout_curve = compute_robometer_curve_signal
compute_vlac_rollout_curve = compute_vlac_curve_signal
