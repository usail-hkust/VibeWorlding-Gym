#!/usr/bin/env python3
"""
根据 wandb 上的 val reward 指标，只保留 top-K 个 checkpoint，删除其余。

用法:
    # dry run（只看不删）
    python cleanup_ckpt_by_reward.py \
        --save_dir ./models/xxx-RL \
        --project MapGenRL \
        --experiment 20260414_qwen3_vl_8b_sft_v5_reward_filtered_grpo \
        --top_k 3 --dry_run

    # 真正删除
    python cleanup_ckpt_by_reward.py \
        --save_dir ./models/xxx-RL \
        --project MapGenRL \
        --experiment 20260414_qwen3_vl_8b_sft_v5_reward_filtered_grpo \
        --top_k 3
"""

import argparse
import os
import re
import shutil
from pathlib import Path

import wandb


def get_step_reward_from_wandb(project: str, experiment: str, metric_key: str) -> dict[int, float]:
    """从 wandb 获取每个 step 的 val reward。"""
    api = wandb.Api()

    # 查找匹配的 run（取最新的一个）
    runs = api.runs(
        project,
        filters={"display_name": experiment},
        order="-created_at",
    )
    runs_list = list(runs)
    if not runs_list:
        # 如果按 display_name 找不到，尝试用 config.experiment_name
        runs = api.runs(
            project,
            filters={"config.experiment_name": experiment},
            order="-created_at",
        )
        runs_list = list(runs)

    if not runs_list:
        raise RuntimeError(
            f"在 wandb 项目 '{project}' 中找不到 experiment '{experiment}'.\n"
            f"请检查 --project 和 --experiment 参数。"
        )

    run = runs_list[0]
    print(f"[wandb] 找到 run: {run.name} (id={run.id}, state={run.state})")

    # 拉取 history，只取需要的列
    step_reward = {}
    for row in run.scan_history(keys=["_step", metric_key]):
        step = row.get("_step")
        reward = row.get(metric_key)
        if step is not None and reward is not None:
            step_reward[int(step)] = float(reward)

    print(f"[wandb] 获取到 {len(step_reward)} 条 step→reward 记录")
    return step_reward


def scan_checkpoints(save_dir: str) -> dict[int, str]:
    """扫描 save_dir 下的 global_step_* 目录，返回 {step: path}。"""
    ckpt_map = {}
    save_path = Path(save_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"保存目录不存在: {save_dir}")

    for d in save_path.iterdir():
        if d.is_dir():
            m = re.match(r"global_step_(\d+)$", d.name)
            if m:
                step = int(m.group(1))
                ckpt_map[step] = str(d)

    print(f"[scan] 找到 {len(ckpt_map)} 个 checkpoint: steps={sorted(ckpt_map.keys())}")
    return ckpt_map


def cleanup(
    save_dir: str,
    project: str,
    experiment: str,
    metric_key: str,
    top_k: int,
    dry_run: bool,
):
    # 1. 获取 reward
    step_reward = get_step_reward_from_wandb(project, experiment, metric_key)

    # 2. 扫描 checkpoint
    ckpt_map = scan_checkpoints(save_dir)

    if not ckpt_map:
        print("[done] 没有找到任何 checkpoint，退出。")
        return

    # 3. 匹配 checkpoint step 到 reward
    matched = []
    unmatched = []
    for step, path in sorted(ckpt_map.items()):
        if step in step_reward:
            matched.append((step, step_reward[step], path))
        else:
            unmatched.append((step, path))

    if unmatched:
        print(f"\n[warn] {len(unmatched)} 个 checkpoint 没有对应的 val reward（可能 test_freq 未对齐）:")
        for step, path in unmatched:
            print(f"  step={step}  {path}")

    if not matched:
        print("[done] 没有任何 checkpoint 能匹配到 val reward，退出。不删除任何文件。")
        return

    # 4. 按 reward 降序排序
    matched.sort(key=lambda x: x[1], reverse=True)

    keep = matched[:top_k]
    remove = matched[top_k:]

    print(f"\n{'=' * 60}")
    print(f"  保留 top-{top_k} checkpoint (按 val reward 降序)")
    print(f"{'=' * 60}")
    for rank, (step, reward, path) in enumerate(keep, 1):
        print(f"  #{rank}  step={step:<6d}  reward={reward:.4f}  {path}")

    if remove:
        print(f"\n{'=' * 60}")
        print(f"  待删除 {len(remove)} 个 checkpoint")
        print(f"{'=' * 60}")
        for step, reward, path in remove:
            print(f"  step={step:<6d}  reward={reward:.4f}  {path}")

    # 对于没有 reward 的 checkpoint，也列为待删除（保守起见也保留，让用户自己决定）
    if unmatched:
        print(f"\n  (另有 {len(unmatched)} 个无 reward 的 checkpoint 不做处理，需手动清理)")

    # 5. 执行删除
    if dry_run:
        print(f"\n[dry_run] 以上为模拟结果，未执行任何删除。去掉 --dry_run 参数以实际删除。")
        return

    freed_bytes = 0
    for step, reward, path in remove:
        # 计算目录大小
        dir_size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        freed_bytes += dir_size
        shutil.rmtree(path)
        print(f"  [deleted] step={step} ({dir_size / 1024**3:.1f} GB)")

    print(f"\n[cleanup] 删除完成，共释放 {freed_bytes / 1024**3:.1f} GB")

    # 6. 更新 latest_checkpointed_iteration.txt 指向最佳 checkpoint
    best_step = keep[0][0]
    latest_file = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
    with open(latest_file, "w") as f:
        f.write(str(best_step))
    print(f"[update] latest_checkpointed_iteration.txt → {best_step}")

    print(f"\n[done] 保留了 {len(keep)} 个 checkpoint，最佳 step={best_step} reward={keep[0][1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="按 val reward 保留 top-K checkpoint，删除其余")
    parser.add_argument("--save_dir", required=True, help="checkpoint 保存根目录 (包含 global_step_* 子目录)")
    parser.add_argument("--project", required=True, help="wandb project name (如 MapGenRL)")
    parser.add_argument("--experiment", required=True, help="wandb experiment/run name")
    parser.add_argument("--metric_key", default="val-core/map_gen_rl/reward/mean@1",
                        help="wandb 中 val reward 的 metric key")
    parser.add_argument("--top_k", type=int, default=3, help="保留 reward 最高的 K 个 checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="只打印，不实际删除")
    args = parser.parse_args()

    cleanup(
        save_dir=args.save_dir,
        project=args.project,
        experiment=args.experiment,
        metric_key=args.metric_key,
        top_k=args.top_k,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
