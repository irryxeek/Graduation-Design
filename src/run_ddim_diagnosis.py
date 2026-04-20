"""统一口径运行 DDIM 步数与 eta 诊断。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import PAPER_MODEL_PATH, PAPER_PROCESSED_DIR, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="运行 DDIM 步数与 eta 诊断")
    parser.add_argument("--model_path", type=str, default=PAPER_MODEL_PATH)
    parser.add_argument("--data_dir", type=str, default=PAPER_PROCESSED_DIR)
    parser.add_argument("--model_type", choices=["legacy", "enhanced"], default="enhanced")
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--metric_space", choices=["standardized", "physical"], default="physical")
    parser.add_argument("--save_root", type=str, default=os.path.join(PROJECT_ROOT, "experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--configs",
        type=str,
        default="ddpm_1000,ddim_50_eta0.0,ddim_100_eta0.0,ddim_200_eta0.0,ddim_100_eta0.5",
    )
    return parser.parse_args()


def run_eval(config_name, save_dir, sampler, ddim_steps, ddim_eta, args):
    from evaluate import main as evaluate_main

    namespace = SimpleNamespace(
        model_path=args.model_path,
        model_type=args.model_type,
        sampler=sampler,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        n_samples=0,
        batch_size=args.batch_size,
        out_channels=args.out_channels,
        data_dir=args.data_dir,
        save_dir=save_dir,
        seed=args.seed,
        metric_space=args.metric_space,
        no_smooth=False,
        smooth=True,
        height_bands="0-5,5-20,20-60",
        pressure_log_transformed="auto",
    )
    print(f"\n=== 运行 {config_name} ===")
    evaluate_main(namespace)


def main():
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root_dir = os.path.join(args.save_root, f"ddim_diagnosis_{timestamp}")
    os.makedirs(root_dir, exist_ok=True)

    all_configs = [
        {"name": "ddpm_1000", "sampler": "ddpm", "ddim_steps": 50, "ddim_eta": 0.0},
        {"name": "ddim_50_eta0.0", "sampler": "ddim", "ddim_steps": 50, "ddim_eta": 0.0},
        {"name": "ddim_100_eta0.0", "sampler": "ddim", "ddim_steps": 100, "ddim_eta": 0.0},
        {"name": "ddim_200_eta0.0", "sampler": "ddim", "ddim_steps": 200, "ddim_eta": 0.0},
        {"name": "ddim_100_eta0.5", "sampler": "ddim", "ddim_steps": 100, "ddim_eta": 0.5},
    ]
    selected_names = {name.strip() for name in args.configs.split(",") if name.strip()}
    configs = [cfg for cfg in all_configs if cfg["name"] in selected_names]

    summary = {}
    for cfg in configs:
        save_dir = os.path.join(root_dir, cfg["name"])
        run_eval(
            config_name=cfg["name"],
            save_dir=save_dir,
            sampler=cfg["sampler"],
            ddim_steps=cfg["ddim_steps"],
            ddim_eta=cfg["ddim_eta"],
            args=args,
        )

        report_path = os.path.join(save_dir, "evaluation_report.json")
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        summary[cfg["name"]] = {
            "metadata": report.get("metadata", {}),
            "summary": report.get("summary", {}),
            "height_band_summary": report.get("height_band_summary", {}),
        }

    summary_path = os.path.join(root_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nDDIM 诊断汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
