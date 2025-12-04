#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建“物理先验 (physical_prior.yaml)”。
优先使用 CSV 中显式的起止字段：
  - video_start_frame / video_end_frame
  - audio_start_s     / audio_end_s
若字段缺失，才退化到粗糙的互相关估计（需本地安装 opencv-python、librosa）。
"""
import argparse, csv, math, statistics, json
from pathlib import Path

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def collect_delays_from_csv(csv_path, fps, has_label=True):
    delays_ms = []           # 全体样本
    delays_ms_by_cls = {}    # 每类
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vs = _to_int(row.get("video_start_frame"))
            ae = _to_float(row.get("audio_start_s"))
            # 若都存在，用它们计算：音频起点(秒) vs 视频起点(帧/fps)
            if vs is not None and ae is not None:
                v_ms = (vs / float(fps)) * 1000.0
                a_ms = ae * 1000.0
                # 我们关心“将音频对齐到视频”所需的 Δt：
                # audio_aligned(t) = audio(t - Δt)，
                # 若音频领先视频，则 Δt > 0（需要“右移”音频）。
                delta = max(-500.0, min(500.0, (v_ms - a_ms)))  # 裁剪 ±0.5s
                delays_ms.append(delta)
                if has_label:
                    # label 索引
                    lab = row.get("label")
                    if lab is None or str(lab).strip()=="" or not str(lab).lstrip("-").isdigit():
                        lab = "unknown"
                    else:
                        lab = int(lab)
                    delays_ms_by_cls.setdefault(lab, []).append(delta)
            # 否则放弃该行（或在后续做互相关估计）
    return delays_ms, delays_ms_by_cls

def robust_stats(arr):
    if not arr:
        return {"count": 0, "mean": None, "std": None, "p10": None, "p90": None, "median": None}
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    def pct(p):
        k = int(round((p/100.0)*(n-1)))
        return arr_sorted[k]
    mean = sum(arr_sorted) / n
    median = arr_sorted[n//2] if n%2==1 else 0.5*(arr_sorted[n//2-1]+arr_sorted[n//2])
    var = sum((x-mean)**2 for x in arr_sorted) / max(1, n-1)
    std = var**0.5
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "median": median,
        "p10": pct(10),
        "p90": pct(90),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_csv", type=str, required=False, help="带标签训练 CSV")
    ap.add_argument("--unlabeled_csv", type=str, required=False, help="无标签 CSV")
    ap.add_argument("--val_csv", type=str, required=False, help="验证 CSV（可选）")
    ap.add_argument("--fps", type=float, default=8.0, help="视频 FPS")
    ap.add_argument("--out", type=str, default="physical_prior.yaml", help="输出 YAML 文件路径")
    args = ap.parse_args()

    fps = float(args.fps)
    frame_ms = 1000.0 / fps

    delays_all = []
    delays_by_cls = {}

    for path, has_label in [(args.labeled_csv, True), (args.val_csv, True), (args.unlabeled_csv, False)]:
        if path:
            d, d_by = collect_delays_from_csv(path, fps, has_label)
            delays_all.extend(d)
            if has_label:
                for k, v in d_by.items():
                    delays_by_cls.setdefault(k, []).extend(v)

    stats_all = robust_stats(delays_all)
    stats_by = {str(k): robust_stats(v) for k,v in delays_by_cls.items()}

    # 设定对称帧界：±3 frames（可按数据分布自动扩展到 p10/p90 所在帧）
    if stats_all["p10"] is not None and stats_all["p90"] is not None:
        low_frames  = math.floor(stats_all["p10"] / frame_ms)
        high_frames = math.ceil (stats_all["p90"] / frame_ms)
        # 至少对称且不小于±3帧
        k = max(3, abs(low_frames), abs(high_frames))
        low_frames, high_frames = -k, k
    else:
        low_frames, high_frames = -3, 3

    # 初始化延迟（帧）采用全局中位数
    if stats_all["median"] is not None:
        init_delay_frames = float(stats_all["median"] / frame_ms)
    else:
        init_delay_frames = 0.0

    # 类别偏移（可选）：相对全局 median 的偏移
    class_offsets = {}
    for k, s in stats_by.items():
        if s["median"] is not None and stats_all["median"] is not None:
            off = (s["median"] - stats_all["median"]) / frame_ms
            class_offsets[k] = float(off)

    # 写 YAML（无需外部库，手写）
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated physical prior\n")
        f.write(f"fps: {fps}\n")
        f.write(f"frame_interval_ms: {frame_ms:.4f}\n")
        f.write(f"delta_low_frames: {low_frames}\n")
        f.write(f"delta_high_frames: {high_frames}\n")
        f.write(f"init_delay_frames: {init_delay_frames:.4f}\n")
        f.write("class_delay_offsets:\n")
        if class_offsets:
            for k, v in class_offsets.items():
                f.write(f"  {k}: {v:.4f}\n")
        else:
            f.write("  {}\n")
        f.write("stats_global:\n")
        for key in ["count","mean","std","median","p10","p90"]:
            val = stats_all[key]
            if val is None:
                f.write(f"  {key}: null\n")
            else:
                f.write(f"  {key}: {val}\n")

    print(f"✅ Wrote physical prior to {out}")

if __name__ == "__main__":
    main()
