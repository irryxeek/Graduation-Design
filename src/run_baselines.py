"""训练并评估轻量判别式基线。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import DEVICE, PAPER_PROCESSED_DIR, PROJECT_ROOT
from ro_retrieval.evaluation.metrics import EvaluationReport
from ro_retrieval.model.baselines import build_baseline
from ro_retrieval.stats_utils import load_stats_from_dir


def parse_args():
    parser = argparse.ArgumentParser(description="训练判别式基线并输出统一评估结果")
    parser.add_argument("--data_dir", type=str, default=PAPER_PROCESSED_DIR)
    parser.add_argument("--models", type=str, default="mlp,cnn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_root", type=str, default=os.path.join(PROJECT_ROOT, "experiments"))
    parser.add_argument("--height_bands", type=str, default="0-5,5-20,20-60")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_height_bands(spec, max_height=60.0):
    bands = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        low_str, high_str = item.split("-", 1)
        low = float(low_str)
        high = float(high_str)
        bands.append((f"{low:g}-{high:g}km", low, high))
    if not bands:
        bands.append((f"0-{max_height:g}km", 0.0, max_height))
    return bands


def restore_physical(values, stats):
    restored = values * stats["y_std"].reshape(1, -1, 1) + stats["y_mean"].reshape(1, -1, 1)
    if stats.get("pressure_log_transformed", False) and restored.shape[1] >= 2:
        restored[:, 1, :] = np.power(10.0, restored[:, 1, :], dtype=np.float32)
    return restored


def compute_height_band_summary(preds, truths, heights, var_names, band_specs):
    summary = {}
    for label, low, high in band_specs:
        mask = (heights >= low) & (heights <= high if np.isclose(high, heights.max()) else heights < high)
        if not np.any(mask):
            continue
        band_result = {}
        for var_idx, var_name in enumerate(var_names):
            pred_band = preds[:, var_idx, :][:, mask].reshape(-1)
            truth_band = truths[:, var_idx, :][:, mask].reshape(-1)
            valid = np.isfinite(pred_band) & np.isfinite(truth_band)
            if valid.sum() == 0:
                continue
            pred_valid = pred_band[valid]
            truth_valid = truth_band[valid]
            rmse = float(np.sqrt(np.mean((pred_valid - truth_valid) ** 2)))
            bias = float(np.mean(pred_valid - truth_valid))
            if np.std(pred_valid) > 1e-10 and np.std(truth_valid) > 1e-10:
                cc = float(np.corrcoef(pred_valid, truth_valid)[0, 1])
            else:
                cc = float("nan")
            band_result[var_name] = {
                "rmse": rmse,
                "bias": bias,
                "cc": cc,
                "n_values": int(valid.sum()),
                "n_levels": int(mask.sum()),
            }
        summary[label] = band_result
    return summary


def evaluate_predictions(preds, truths, heights, band_specs):
    var_names = ["temperature", "pressure", "humidity"][:preds.shape[1]]
    report = EvaluationReport(variable_names=var_names)
    for idx in range(len(preds)):
        report.add_sample(pred=preds[idx], truth=truths[idx], sample_idx=idx)
    return {
        "summary": report.summary(),
        "height_band_summary": compute_height_band_summary(preds, truths, heights, var_names, band_specs),
    }


def run_model_inference(model, loader, device):
    outputs = []
    targets = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy()
            outputs.append(pred)
            targets.append(y_batch.numpy())
    return np.concatenate(outputs, axis=0), np.concatenate(targets, axis=0)


def train_single_model(model_name, args, timestamp):
    data_dir = args.data_dir
    train_x = np.load(os.path.join(data_dir, "train_x.npy")).astype(np.float32)
    train_y = np.load(os.path.join(data_dir, "train_y.npy")).astype(np.float32)
    val_x = np.load(os.path.join(data_dir, "val_x.npy")).astype(np.float32)
    val_y = np.load(os.path.join(data_dir, "val_y.npy")).astype(np.float32)
    test_x = np.load(os.path.join(data_dir, "test_x.npy")).astype(np.float32)
    test_y = np.load(os.path.join(data_dir, "test_y.npy")).astype(np.float32)
    stats = load_stats_from_dir(data_dir)
    heights = np.asarray(stats["target_heights"], dtype=np.float32)
    band_specs = parse_height_bands(args.height_bands, max_height=float(heights[-1]))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_baseline(model_name, input_length=train_x.shape[-1], out_channels=train_y.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    save_dir = os.path.join(args.save_root, f"baseline_{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"{model_name}_best.pth")
    log_path = os.path.join(save_dir, "training_log.json")

    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history = {"train_loss": [], "val_loss": []}

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                pred = model(x_batch)
                val_losses.append(criterion(pred, y_batch).item())

        train_loss = float(np.mean(epoch_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[{model_name}] epoch {epoch + 1}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            patience_left = args.patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    preds_std, truths_std = run_model_inference(model, test_loader, DEVICE)
    preds_phys = restore_physical(preds_std, stats)
    truths_phys = restore_physical(truths_std, stats)

    standardized_eval = evaluate_predictions(preds_std, truths_std, heights, band_specs)
    physical_eval = evaluate_predictions(preds_phys, truths_phys, heights, band_specs)

    payload = {
        "metadata": {
            "model_name": model_name,
            "data_dir": os.path.relpath(data_dir, PROJECT_ROOT) if os.path.isabs(data_dir) else data_dir,
            "epochs_requested": args.epochs,
            "epochs_trained": len(history["train_loss"]),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
        "standardized": standardized_eval,
        "physical": physical_eval,
        "units": {
            "temperature": {"standardized": "standardized", "physical": "K"},
            "pressure": {"standardized": "standardized", "physical": "hPa"},
            "humidity": {"standardized": "standardized", "physical": "g/kg"},
        },
        "history": history,
        "runtime_minutes": (time.time() - start_time) / 60.0,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return save_dir, payload


def main():
    args = parse_args()
    set_seed(args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    summary = {}
    for model_name in model_names:
        save_dir, payload = train_single_model(model_name, args, timestamp)
        summary[model_name] = {
            "save_dir": os.path.relpath(save_dir, PROJECT_ROOT),
            "best_val_loss": payload["metadata"]["best_val_loss"],
            "standardized": payload["standardized"]["summary"],
            "physical": payload["physical"]["summary"],
        }

    summary_path = os.path.join(args.save_root, f"baseline_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"基线汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
