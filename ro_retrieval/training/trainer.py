"""
系统化训练器
============
特性:
  - 支持 train / val 划分, 训练时实时监控验证损失
  - Early Stopping (patience 轮无改善则停止)
  - 自动保存最佳模型 (best_model.pth) + 定期检查点
  - 训练日志 (loss_history) 保存为 JSON + npy
  - 支持 legacy / enhanced 两种模型
  - 支持单变量 / 多变量模式
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ro_retrieval.config import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    TIMESTEPS, PROCESSED_DIR, PROJECT_ROOT,
)
from ro_retrieval.data.dataset import RODataset, ROMultiVarDataset
from ro_retrieval.model.unet import ConditionalUNet1D, EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule


class Trainer:
    """
    掩星扩散模型系统化训练器

    Usage:
        trainer = Trainer(
            data_dir="Data/Processed",
            model_type="enhanced",
            mode="multi",
        )
        trainer.train()
    """

    def __init__(
        self,
        data_dir: str = PROCESSED_DIR,
        model_type: str = "enhanced",
        mode: str = "multi",
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        save_dir: str = PROJECT_ROOT,
        patience: int = 20,
        save_every: int = 10,
        device=None,
    ):
        """
        Args:
            data_dir: 包含 train_x.npy, train_y.npy (以及可选 val_x.npy, val_y.npy) 的目录
            model_type: "legacy" | "enhanced"
            mode: "single" (仅温度) | "multi" (温度+压力+湿度)
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
            save_dir: 模型保存目录
            patience: Early Stopping 容忍轮数
            save_every: 每隔多少轮保存检查点
            device: 计算设备
        """
        self.data_dir = data_dir
        self.model_type = model_type
        self.mode = mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_dir = save_dir
        self.patience = patience
        self.save_every = save_every
        self.device = device or DEVICE

        # 训练日志
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        # 会在 _setup() 中初始化
        self.model = None
        self.optimizer = None
        self.schedule = None
        self.train_loader = None
        self.val_loader = None
        self.out_channels = 1
        self.model_prefix = "ro_diffusion"

    def _setup(self):
        """初始化数据集、模型、优化器"""

        # ---- 数据 ----
        train_x_path = os.path.join(self.data_dir, "train_x.npy")
        train_y_path = os.path.join(self.data_dir, "train_y.npy")

        if not os.path.exists(train_x_path):
            raise FileNotFoundError(f"训练数据不存在: {train_x_path}")

        # 检查是否已有 val 数据
        val_x_path = os.path.join(self.data_dir, "val_x.npy")
        val_y_path = os.path.join(self.data_dir, "val_y.npy")
        has_val_split = os.path.exists(val_x_path) and os.path.exists(val_y_path)

        if self.mode == "multi":
            train_dataset = ROMultiVarDataset(train_x_path, train_y_path)
            self.out_channels = train_dataset.num_vars
        else:
            train_dataset = RODataset(train_x_path, train_y_path)
            self.out_channels = 1

        if has_val_split:
            print("[Trainer] 检测到 val_x.npy / val_y.npy, 使用预划分验证集")
            if self.mode == "multi":
                val_dataset = ROMultiVarDataset(val_x_path, val_y_path)
            else:
                val_dataset = RODataset(val_x_path, val_y_path)
        else:
            # 自动从训练集按 90/10 划分
            print("[Trainer] 未检测到验证集, 自动从训练集划分 10% 作为验证")
            n_total = len(train_dataset)
            n_val = max(int(n_total * 0.1), 1)
            n_train = n_total - n_val
            train_dataset, val_dataset = random_split(
                train_dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        print(f"[Trainer] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

        # ---- 模型 ----
        if self.model_type == "enhanced":
            self.model = EnhancedConditionalUNet1D(
                in_channels=self.out_channels,
                cond_channels=1,
                out_channels=self.out_channels,
                use_cross_attention=True,
            ).to(self.device)
            self.model_prefix = "enhanced_ro_diffusion"
        else:
            self.model = ConditionalUNet1D(
                in_channels=self.out_channels,
                cond_channels=1,
                out_channels=self.out_channels,
            ).to(self.device)
            self.model_prefix = "ro_diffusion"

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] 模型: {self.model_type}, 参数量: {n_params:,}")
        print(f"[Trainer] 输出通道: {self.out_channels}")

        # ---- 优化器 ----
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.schedule = DiffusionSchedule(TIMESTEPS, device=self.device)

    def _train_one_epoch(self, epoch):
        """单轮训练"""
        self.model.train()
        loss_fn = nn.MSELoss()
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")
        for condition, x_0 in pbar:
            condition = condition.to(self.device)
            x_0 = x_0.to(self.device)
            b = x_0.shape[0]

            t = torch.randint(0, TIMESTEPS, (b, 1), device=self.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.schedule.q_sample(x_0, t, noise)

            noise_pred = self.model(x_t, t, condition)
            loss = loss_fn(noise_pred, noise)

            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return epoch_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self):
        """验证集评估"""
        self.model.eval()
        loss_fn = nn.MSELoss()
        val_loss = 0
        n_batches = 0

        for condition, x_0 in self.val_loader:
            condition = condition.to(self.device)
            x_0 = x_0.to(self.device)
            b = x_0.shape[0]

            t = torch.randint(0, TIMESTEPS, (b, 1), device=self.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.schedule.q_sample(x_0, t, noise)

            noise_pred = self.model(x_t, t, condition)
            loss = loss_fn(noise_pred, noise)

            val_loss += loss.item()
            n_batches += 1

        return val_loss / max(n_batches, 1)

    def train(self):
        """执行完整训练流程"""
        self._setup()
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"开始训练")
        print(f"  设备     : {self.device}")
        print(f"  Epochs   : {self.epochs}")
        print(f"  Batch    : {self.batch_size}")
        print(f"  LR       : {self.lr}")
        print(f"  模式     : {self.mode} ({self.out_channels} channels)")
        print(f"  模型     : {self.model_type}")
        print(f"  Patience : {self.patience}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        for epoch in range(self.epochs):
            # 训练
            train_loss = self._train_one_epoch(epoch)
            self.train_losses.append(train_loss)

            # 验证
            val_loss = self._validate()
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch + 1:3d}/{self.epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"best_val={self.best_val_loss:.6f}")

            # Early Stopping 逻辑
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                best_path = os.path.join(
                    self.save_dir, f"{self.model_prefix}_best.pth"
                )
                torch.save(self.model.state_dict(), best_path)
                print(f"  ✓ 最佳模型已保存: {best_path}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"\n[Early Stopping] 验证损失连续 {self.patience} 轮无改善, 停止训练")
                    break

            # 定期保存检查点
            if (epoch + 1) % self.save_every == 0:
                ckpt_path = os.path.join(
                    self.save_dir, f"{self.model_prefix}_epoch_{epoch + 1}.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  检查点已保存: {ckpt_path}")

        elapsed = time.time() - start_time
        print(f"\n训练完成! 耗时: {elapsed / 60:.1f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")

        # 保存训练日志
        self._save_log()

        return self.model

    def _save_log(self):
        """保存训练日志 (JSON + npy)"""
        log = {
            "model_type": self.model_type,
            "mode": self.mode,
            "out_channels": self.out_channels,
            "epochs_trained": len(self.train_losses),
            "best_val_loss": float(self.best_val_loss),
            "train_losses": [float(x) for x in self.train_losses],
            "val_losses": [float(x) for x in self.val_losses],
            "config": {
                "batch_size": self.batch_size,
                "lr": self.lr,
                "patience": self.patience,
            },
        }

        json_path = os.path.join(self.save_dir, f"{self.model_prefix}_training_log.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        np.save(
            os.path.join(self.save_dir, f"{self.model_prefix}_loss_history.npy"),
            np.array(self.train_losses),
        )

        print(f"训练日志已保存: {json_path}")
