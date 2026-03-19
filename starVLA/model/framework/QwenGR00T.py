# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
Qwen-GR00T 框架
一个基于 Qwen-VL 与 Flow-matching 头的轻量实现，用于直接预测连续动作
Flow-matching 头部来自 GR00T N1.5，保留其版权说明
"""
from starVLA.model.tools import FRAMEWORK_REGISTRY  # 框架注册器，负责按名称实例化/管理框架
from starVLA.training.trainer_utils.trainer_tools import resize_images  # 训练/推理统一的图像尺寸调整
# Flow-Matching 动作头工厂与类型
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.vlm import get_vlm_model  # 构建 VLM（Qwen-VL 等）接口的工厂
from starVLA.model.framework.base_framework import baseframework  # 通用框架基类，封装训练时需要的接口
from starVLA.model.framework.singal_utils import build_hidden_state_save_payload, compute_liv_signal
# 将输入图像安全转成 PIL，保持信息完整
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch  # 初始化日志记录器
from PIL import Image  # 图像容器
import numpy as np  # 数值计算
import torch.nn as nn  # 神经网络模块封装
import torch  # 核心张量与自动求导库
from typing import List, Optional, Tuple  # 类型注解
from tqdm import tqdm  # 进度条工具
from typing import List  # 类型别名（保留原样）
import sys  # 操作 Python 路径
from pathlib import Path  # 文件路径处理
import time  # 简单时间戳，用于生成唯一保存名

# Add workspace root to Python path if not already there
# 确保项目根目录加入 sys.path，方便模块绝对导入
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))


# 创建统一的日志记录器
logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@FRAMEWORK_REGISTRY.register("QwenGR00T")  # 向框架注册表注册本类，便于按字符串加载
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    多模态视觉-语言-动作模型。

    组成模块：
      - Qwen2.5 VL 接口，提供融合后的图文 token 表征
      - 分层 QFormer，用于多层特征聚合
      - DINO 编码器，提取多视角稠密空间 token
      - DiT 扩散头，用于未来动作序列建模

    目标：在图像与指令条件下预测未来的连续动作。
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        构建所有子模块并缓存关键配置值。

        参数：
            config：层级化的配置（OmegaConf/dict），包含框架与训练器字段。
            **kwargs：预留的额外参数（当前未使用）。
        """
        super().__init__()
        self.config = config  # 保存全局配置引用，便于子模块读取
        self.qwen_vl_interface = get_vlm_model(config=self.config)  # 得到VLM类
        # align dims --> we should put them to config or no?
        # 对齐维度，把VLM的隐藏空间维度赋大小给动作头的维度
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(
            config=self.config)  # 修复后续引用 得到FlowMatching类

        self.future_action_window_size = config.framework.action_model.future_action_window_size  # 未来动作窗口长度
        self.past_action_window_size = config.framework.action_model.past_action_window_size  # 过去动作窗口长度
        self.chunk_len = self.past_action_window_size + 1 + \
            self.future_action_window_size  # 总窗口=过去+当前+未来

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        前向训练流程，返回动作损失。
        """
        # 前向训练主流程，输入为一个 batch 的字典列表
        batch_images = [example["image"] for example in examples]  # [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"]
                   for example in examples]  # label [B， len, 7]

        # [B, 1, state_dim]
        state = [example["state"]
                 for example in examples] if "state" in examples[0] else None

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions)
        # 用 bfloat16 自动混合精度提升吞吐
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # VLM进行前向，输入包括图片以及指令，得到每个layer的hidden_states
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            # 取最后一层的hidden state
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        # 动作头使用 float32 以避免数值不稳定
        with torch.autocast("cuda", dtype=torch.float32):
            # 构建一个对应样本时间长度的完整动作序列
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            # 动作序列切分成chunk,从后往前分
            # (B, chunk_len, action_dim)
            actions_target = actions[:, -
                                     (self.future_action_window_size+1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get(
                    "repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4  # 训练配置里设置的重复采样次数
            )
            # 将动作/特征重复多次，用于 flow-matching 多噪声采样
            actions_target_repeated = actions_target.repeat(
                repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(
                repeated_diffusion_steps, 1, 1)

            # 状态（若存在）也需重复扩展
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            inject_signal_train = kwargs.get(
                "inject_signal_train",
                self.config.framework.get(
                    "inject_signal_train",
                    self.config.framework.get("use_signal_train", False),
                ),
            )
            signal_repeated = None
            if inject_signal_train:
                signal = torch.empty(
                    len(examples),
                    device=last_hidden.device,
                    dtype=last_hidden.dtype,
                )
                missing_indices = []
                missing_images = []
                missing_instructions = []
                for idx, example in enumerate(examples):
                    cached_signal = example.get("signal", None)
                    if cached_signal is None:
                        missing_indices.append(idx)
                        missing_images.append(batch_images[idx])
                        missing_instructions.append(instructions[idx])
                        continue
                    signal[idx] = torch.as_tensor(
                        cached_signal,
                        device=last_hidden.device,
                        dtype=last_hidden.dtype,
                    ).reshape(-1)[0]

                if missing_indices:
                    computed_signal = compute_liv_signal(
                        batch_images=missing_images,
                        instructions=missing_instructions,
                        device=last_hidden.device,
                        dtype=last_hidden.dtype,
                    )
                    for missing_offset, example_idx in enumerate(missing_indices):
                        signal[example_idx] = computed_signal[missing_offset]
                signal_repeated = signal.repeat(
                    repeated_diffusion_steps)

            # 交给动作头计算 MSE 形式的 flow-matching 损失
            action_loss = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                signal=signal_repeated,
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        步骤：
          1. 若配置了训练分辨率，则将图像 resize 到训练分辨率
          2. 使用 QwenVL 编码（保留隐藏层）
          6. 返回归一化后的动作轨迹
        返回：
            字典：
                normalized_actions (np.ndarray)：形状 [B, T, action_dim]，表示扩散采样得到的归一化动作
        """
        # 推理接口：输入同样是字典或字典列表
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"])
                        for example in examples]  # [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        # [B, 1, state_dim]
        state = [example["state"]
                 for example in examples] if "state" in examples[0] else None

        # 推理时按训练配置的分辨率对齐图像
        train_obs_image_size = getattr(
            self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(
                batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions)
        # VLM 仍然用 bfloat16 进行编码
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        base_last_hidden = last_hidden
        signal = None
        inject_signal_infer = kwargs.get(
            "inject_signal_infer",
            self.config.framework.get(
                "inject_signal_infer",
                self.config.framework.get("use_signal_infer", False),
            ),
        )
        inject_signal_infer = bool(inject_signal_infer)
        if inject_signal_infer:
            signal = compute_liv_signal(
                batch_images=batch_images,
                instructions=instructions,
                device=last_hidden.device,
                dtype=last_hidden.dtype,
            )

        # 推理时保存最后一层 hidden states 及其多模态元信息，便于离线切分图像/文本 token
        hidden_save_path = kwargs.get("hidden_save_path", None)
        if hidden_save_path:
            save_payload = build_hidden_state_save_payload(
                qwen_inputs=qwen_inputs,
                last_hidden=last_hidden,
                base_last_hidden=base_last_hidden,
                batch_images=batch_images,
                instructions=instructions,
                tokenizer=self.qwen_vl_interface.processor.tokenizer,
                image_token_id=self.qwen_vl_interface.model.config.image_token_id,
            )
            save_payload["signal"] = signal.detach(
            ).cpu() if signal is not None else None
            save_payload["signal_flags"] = {
                "inject_signal_infer": inject_signal_infer,
            }
        if hidden_save_path:
            save_path = Path(hidden_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(save_payload, save_path)

        # 将可选 state 转成张量并放到同一设备
        state = torch.from_numpy(np.array(state)).to(
            last_hidden.device, dtype=last_hidden.dtype) if state is not None else None

        # Step 4: Action Expert Forward
        # 动作头推理用 float32，执行多步采样
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state, signal if inject_signal_infer else None)  # (B, chunk_len, action_dim)
        # 输出 numpy，通常仍处于归一化空间
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    # 调试入口：可单独运行本文件验证模型构建与前向
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()  # 构造命令行解析器
    parser.add_argument("--config_yaml", type=str,
                        default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml", help="Path to YAML config")  # 读取配置路径
    args, clipargs = parser.parse_known_args()  # 解析已知参数，其余保留

    # 远程调试监听，方便 VSCode/IDE attach
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    # 读取配置
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"

    # 构建模型实例
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    # fake sample
    image = Image.fromarray(np.random.randint(
        0, 255, (224, 224, 3), dtype=np.uint8))  # 随机生成一张 224x224 的假图像
    # Create a sample
    sample = {
        # action_chunk, action_dim
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],  # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }
    sample2 = {
        # action_chunk, action_dim
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],  # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    # 拼一个 batch（batch size=2）
    batch = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # 自动选择 GPU/CPU
    model = model.to(device)
    # 前向计算动作损失
    forward_output = model(batch)  # 前向得到损失字典
    action_loss = forward_output['action_loss']  # 取出动作损失张量
    print(f"Action Loss: {action_loss.item()}")  # 打印标量损失

    # test predict action
    # 推理动作序列
    predict_output = model.predict_action(
        examples=[sample])  # , state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']  # 取出动作结果
    print(f"Unnormalized Action: {normalized_actions}")  # 打印动作序列

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data  # 取出 VLA 数据集配置
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # forward model with dataloader
    # 简单迭代一个 batch 验证数据管线
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        # break

    action = model.predict_action(examples=batch)  # 生成动作
    print("Finished")  # 结束示例流程
