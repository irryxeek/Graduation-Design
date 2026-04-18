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
  - 当前默认配置对齐论文 ATP+WAP 主线
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
    TIMESTEPS, PAPER_EPOCHS, PAPER_HUMIDITY_GRAD_WEIGHT,
    PAPER_MONITOR_TARGET, PAPER_PATIENCE, PAPER_PROCESSED_DIR,
    PAPER_VAR_WEIGHTS, PROJECT_ROOT,
)
from ro_retrieval.data.dataset import RODataset, ROMultiVarDataset
from ro_retrieval.model.unet import ConditionalUNet1D, EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule


class Trainer:
    """
    掩星扩散模型系统化训练器

    Usage:
        trainer = Trainer(
            data_dir="Data/Processed_ATP_WAP_2025",
            model_type="enhanced",
            mode="multi",
        )
        trainer.train()
    """

    def __init__(
        self,
        data_dir: str = PAPER_PROCESSED_DIR,
        model_type: str = "enhanced",
        mode: str = "multi",
        epochs: int = PAPER_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        save_dir: str = PROJECT_ROOT,
        patience: int = PAPER_PATIENCE,
        save_every: int = 10,
        device=None,
        var_weights: list = None,
        monitor_target: str = PAPER_MONITOR_TARGET,
        humidity_grad_weight: float = PAPER_HUMIDITY_GRAD_WEIGHT,
        humidity_cc_weight: float = 0.0,
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
            var_weights: 各变量的损失权重 [T, P, Q], 默认按论文主线使用 [1.0, 1.0, 4.0]
            monitor_target: Early Stopping 监控目标;
                loss/temperature/pressure/humidity/humidity_cc
            humidity_grad_weight: 湿度廓线梯度约束权重, 默认按论文主线使用 0.05
            humidity_cc_weight: 湿度相关性损失权重, 0 表示关闭
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
        self.var_weights = var_weights or list(PAPER_VAR_WEIGHTS)
        self.monitor_target = monitor_target
        self.humidity_grad_weight = humidity_grad_weight
        self.humidity_cc_weight = humidity_cc_weight

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
        self.variable_names = ["temperature", "pressure", "humidity"]
        self.best_monitor_value = None

    def _get_channel_weights(self):
        """返回按输出通道裁剪/补齐后的损失权重张量。"""
        if self.mode != "multi" or self.out_channels <= 1:
            return None

        weights = list(self.var_weights)
        if len(weights) < self.out_channels:
            weights.extend([weights[-1]] * (self.out_channels - len(weights)))

        return torch.tensor(
            weights[:self.out_channels],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1, 1)

    def _resolve_monitor_target(self):
        """解析当前监控目标。"""
        if self.monitor_target == "loss":
            return "loss"
        if self.monitor_target == "humidity_cc":
            return "humidity_cc"

        available = self.variable_names[:self.out_channels]
        if self.monitor_target not in available:
            raise ValueError(
                f"monitor_target={self.monitor_target} 不可用, "
                f"当前输出变量为: {available}"
            )
        return self.monitor_target

    def _is_monitor_improved(self, monitor_target, monitor_value):
        """判断监控指标是否改善。"""
        if self.best_monitor_value is None:
            return True
        if monitor_target == "humidity_cc":
            return monitor_value > self.best_monitor_value
        return monitor_value < self.best_monitor_value

    def _predict_x0(self, x_t, t, noise_pred):
        """根据噪声预测反推 x0。"""
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return (x_t - sqrt_one_minus_alpha * noise_pred) / (sqrt_alpha + 1e-8)

    def _batch_profile_correlation(self, pred, truth):
        """计算 batch 内逐样本廓线相关系数均值。"""
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        truth_centered = truth - truth.mean(dim=-1, keepdim=True)
        numerator = (pred_centered * truth_centered).mean(dim=-1)
        denominator = (
            pred_centered.pow(2).mean(dim=-1).sqrt()
            * truth_centered.pow(2).mean(dim=-1).sqrt()
            + 1e-8
        )
        cc = numerator / denominator
        cc = torch.clamp(cc, -1.0, 1.0)
        return cc.mean()

    def _compute_loss(self, noise_pred, noise, x_t, t, x_0, weights):
        """统一计算训练/验证损失及附加诊断项。"""
        sq_error = (noise_pred - noise) ** 2
        if weights is not None:
            base_loss = torch.mean(weights * sq_error)
        else:
            base_loss = torch.mean(sq_error)

        humidity_grad_loss = torch.tensor(0.0, device=noise_pred.device)
        humidity_cc_loss = torch.tensor(0.0, device=noise_pred.device)
        humidity_cc_value = torch.tensor(0.0, device=noise_pred.device)
        pred_x0 = None
        if (
            (self.humidity_grad_weight > 0 or self.humidity_cc_weight > 0)
            and self.mode == "multi"
            and self.out_channels >= 3
        ):
            pred_x0 = self._predict_x0(x_t, t, noise_pred)
        if (
            self.humidity_grad_weight > 0
            and pred_x0 is not None
        ):
            pred_h = pred_x0[:, 2, :]
            true_h = x_0[:, 2, :]
            pred_grad = pred_h[:, 1:] - pred_h[:, :-1]
            true_grad = true_h[:, 1:] - true_h[:, :-1]
            humidity_grad_loss = nn.functional.mse_loss(pred_grad, true_grad)
        if (
            self.humidity_cc_weight > 0
            and pred_x0 is not None
        ):
            pred_h = pred_x0[:, 2, :]
            true_h = x_0[:, 2, :]
            humidity_cc_value = self._batch_profile_correlation(pred_h, true_h)
            humidity_cc_loss = 1.0 - humidity_cc_value

        total_loss = (
            base_loss
            + self.humidity_grad_weight * humidity_grad_loss
            + self.humidity_cc_weight * humidity_cc_loss
        )
        return total_loss, {
            "base_loss": float(base_loss.detach().item()),
            "humidity_grad_loss": float(humidity_grad_loss.detach().item()),
            "humidity_cc_loss": float(humidity_cc_loss.detach().item()),
            "humidity_cc_value": float(humidity_cc_value.detach().item()),
        }

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
        epoch_loss = 0
        n_batches = 0

        weights = self._get_channel_weights()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")
        for condition, x_0 in pbar:
            condition = condition.to(self.device)
            x_0 = x_0.to(self.device)
            b = x_0.shape[0]

            t = torch.randint(0, TIMESTEPS, (b, 1), device=self.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.schedule.q_sample(x_0, t, noise)

            noise_pred = self.model(x_t, t, condition)
            loss, loss_info = self._compute_loss(
                noise_pred=noise_pred,
                noise=noise,
                x_t=x_t,
                t=t,
                x_0=x_0,
                weights=weights,
            )

            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            postfix = {"loss": f"{loss.item():.6f}"}
            if self.humidity_grad_weight > 0 and self.out_channels >= 3:
                postfix["q_grad"] = f"{loss_info['humidity_grad_loss']:.6f}"
            pbar.set_postfix(postfix)

        return epoch_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self):
        """验证集评估"""
        self.model.eval()
        val_loss = 0
        n_batches = 0
        weights = self._get_channel_weights()
        per_var_loss = torch.zeros(self.out_channels, dtype=torch.float64)
        humidity_grad_loss_sum = 0.0
        humidity_cc_sum = 0.0
        humidity_cc_loss_sum = 0.0

        for condition, x_0 in self.val_loader:
            condition = condition.to(self.device)
            x_0 = x_0.to(self.device)
            b = x_0.shape[0]

            t = torch.randint(0, TIMESTEPS, (b, 1), device=self.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.schedule.q_sample(x_0, t, noise)

            noise_pred = self.model(x_t, t, condition)
            channel_mse = ((noise_pred - noise) ** 2).mean(dim=(0, 2))
            per_var_loss += channel_mse.detach().cpu().double()
            loss, loss_info = self._compute_loss(
                noise_pred=noise_pred,
                noise=noise,
                x_t=x_t,
                t=t,
                x_0=x_0,
                weights=weights,
            )
            humidity_grad_loss_sum += loss_info["humidity_grad_loss"]
            humidity_cc_sum += loss_info["humidity_cc_value"]
            humidity_cc_loss_sum += loss_info["humidity_cc_loss"]

            val_loss += loss.item()
            n_batches += 1

        avg_val_loss = val_loss / max(n_batches, 1)
        avg_per_var_loss = (per_var_loss / max(n_batches, 1)).tolist()
        return {
            "loss": avg_val_loss,
            "humidity_grad_loss": humidity_grad_loss_sum / max(n_batches, 1),
            "humidity_cc": humidity_cc_sum / max(n_batches, 1),
            "humidity_cc_loss": humidity_cc_loss_sum / max(n_batches, 1),
            "per_var_loss": {
                self.variable_names[i]: float(avg_per_var_loss[i])
                for i in range(self.out_channels)
            },
        }

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
        monitor_target = self._resolve_monitor_target()

        for epoch in range(self.epochs):
            # 训练
            train_loss = self._train_one_epoch(epoch)
            self.train_losses.append(train_loss)

            # 验证
            val_metrics = self._validate()
            val_loss = val_metrics["loss"]
            if monitor_target == "loss":
                monitor_value = val_loss
            elif monitor_target == "humidity_cc":
                monitor_value = val_metrics["humidity_cc"]
            else:
                monitor_value = val_metrics["per_var_loss"][monitor_target]
            self.val_losses.append(val_loss)

            best_monitor_str = (
                f"{self.best_monitor_value:.6f}"
                if self.best_monitor_value is not None
                else "N/A"
            )
            print(f"Epoch {epoch + 1:3d}/{self.epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"monitor({monitor_target})={monitor_value:.6f}  "
                  f"best_monitor={best_monitor_str}")
            if self.out_channels > 1:
                per_var_str = "  ".join(
                    f"{name}={value:.6f}"
                    for name, value in val_metrics["per_var_loss"].items()
                )
                extra = ""
                if self.humidity_grad_weight > 0 and self.out_channels >= 3:
                    extra = f"  q_grad={val_metrics['humidity_grad_loss']:.6f}"
                if self.out_channels >= 3:
                    extra += f"  q_cc={val_metrics['humidity_cc']:.6f}"
                print(f"  [Val per-var] {per_var_str}{extra}")

            # Early Stopping 逻辑
            if self._is_monitor_improved(monitor_target, monitor_value):
                self.best_monitor_value = monitor_value
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                best_path = os.path.join(
                    self.save_dir, f"{self.model_prefix}_best.pth"
                )
                torch.save(self.model.state_dict(), best_path)
                print(
                    f"  [BEST] 最佳模型已保存: {best_path} "
                    f"(monitor={monitor_target}:{monitor_value:.6f})"
                )
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(
                        f"\n[Early Stopping] 监控指标 {monitor_target} 连续 "
                        f"{self.patience} 轮无改善, 停止训练"
                    )
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
        print(f"最佳监控指标({monitor_target}): {self.best_monitor_value:.6f}")

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
            "best_monitor_value": float(self.best_monitor_value),
            "train_losses": [float(x) for x in self.train_losses],
            "val_losses": [float(x) for x in self.val_losses],
            "config": {
                "batch_size": self.batch_size,
                "lr": self.lr,
                "patience": self.patience,
                "var_weights": [float(x) for x in self.var_weights],
                "monitor_target": self.monitor_target,
                "humidity_grad_weight": self.humidity_grad_weight,
                "humidity_cc_weight": self.humidity_cc_weight,
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

    @torch.no_grad()
    def evaluate_test(self, model_path: str = None, num_samples: int = 5):
        """
        在独立测试集上评估模型性能。

        当前实现直接复用 `src.evaluate` 的论文主线评估口径，
        避免训练器内部再维护一套偏离主实验结果的逻辑。

        Args:
            model_path: 模型权重路径，默认使用 best 模型
            num_samples: 评估样本数；<=0 表示评估全部测试集

        Returns:
            dict: 包含评估摘要和结果文件路径的字典
        """
        if self.model is None:
            self._setup()

        if model_path is None:
            model_path = os.path.join(self.save_dir, f"{self.model_prefix}_best.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        from types import SimpleNamespace
        from src.evaluate import main as evaluate_main

        os.makedirs(self.save_dir, exist_ok=True)
        eval_save_dir = os.path.join(self.save_dir, f"{self.model_prefix}_test_eval")
        evaluate_main(
            SimpleNamespace(
                model_path=model_path,
                model_type=self.model_type,
                sampler="ddim",
                ddim_steps=50,
                n_samples=0 if num_samples <= 0 else num_samples,
                batch_size=self.batch_size,
                out_channels=self.out_channels,
                data_dir=self.data_dir,
                save_dir=eval_save_dir,
                seed=42,
                metric_space="standardized",
                smooth=True,
                no_smooth=False,
            )
        )

        report_path = os.path.join(eval_save_dir, "evaluation_report.json")
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        result = {
            "model_path": model_path,
            "test_samples": report.get("n_samples"),
            "evaluation_report_path": report_path,
            "summary": report.get("summary", {}),
            "metadata": report.get("metadata", {}),
        }

        result_path = os.path.join(self.save_dir, f"{self.model_prefix}_test_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存: {result_path}")

        return result

    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> dict:
        """
        计算评估指标

        Args:
            preds: 预测值 (N, C, L) 或 (N, 1, L)
            targets: 真实值，同上

        Returns:
            dict: 各项指标
        """
        # 展平计算整体指标
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()

        mse = np.mean((preds_flat - targets_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_flat - targets_flat))

        # R² 和相关系数（按样本计算后取平均）
        r2_list = []
        corr_list = []
        for i in range(len(preds)):
            p = preds[i].flatten()
            t = targets[i].flatten()

            # R²
            ss_res = np.sum((t - p) ** 2)
            ss_tot = np.sum((t - np.mean(t)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            r2_list.append(r2)

            # Pearson 相关系数
            corr = np.corrcoef(p, t)[0, 1]
            corr_list.append(corr if not np.isnan(corr) else 0)

        r2_mean = np.mean(r2_list)
        corr_mean = np.mean(corr_list)

        # 各变量的详细指标（多变量模式）
        rmse_per_var = []
        mae_per_var = []
        r2_per_var = []
        corr_per_var = []

        if preds.ndim == 3 and preds.shape[1] > 1:
            for c in range(preds.shape[1]):
                p_var = preds[:, c, :].flatten()
                t_var = targets[:, c, :].flatten()

                # RMSE
                var_mse = np.mean((p_var - t_var) ** 2)
                rmse_per_var.append(np.sqrt(var_mse))

                # MAE
                mae_per_var.append(np.mean(np.abs(p_var - t_var)))

                # R² (整体)
                ss_res = np.sum((t_var - p_var) ** 2)
                ss_tot = np.sum((t_var - np.mean(t_var)) ** 2)
                r2_per_var.append(1 - ss_res / (ss_tot + 1e-8))

                # 相关系数
                corr = np.corrcoef(p_var, t_var)[0, 1]
                corr_per_var.append(corr if not np.isnan(corr) else 0)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_mean": r2_mean,
            "corr_mean": corr_mean,
            "rmse_per_var": rmse_per_var,
            "mae_per_var": mae_per_var,
            "r2_per_var": r2_per_var,
            "corr_per_var": corr_per_var,
            "r2_per_sample": r2_list,
            "corr_per_sample": corr_list,
        }
