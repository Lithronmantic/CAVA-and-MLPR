import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob
import cv2
import librosa
from tqdm import tqdm


# 复用之前的 Dataset 类 (精简版)
def extract_signals(video_path, audio_path):
    # --- Video ---
    cap = cv2.VideoCapture(video_path)
    visual_energy = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        visual_energy.append(np.mean(gray))
    cap.release()
    v = np.array(visual_energy)
    if len(v) < 10: return None, None
    if np.std(v) > 1e-5: v = (v - np.mean(v)) / np.std(v)  # Normalize

    # --- Audio ---
    y, sr = librosa.load(audio_path, sr=None)
    samples_per_frame = int(len(y) / len(v))
    if samples_per_frame == 0: return None, None
    a = librosa.feature.rms(y=y, frame_length=2048, hop_length=samples_per_frame, center=True)[0]

    min_len = min(len(v), len(a))
    v = v[:min_len]
    a = a[:min_len]
    if np.std(a) > 1e-5: a = (a - np.mean(a)) / np.std(a)  # Normalize

    return v, a


def debug_negative_lags(dataset_root):
    video_files = glob.glob(os.path.join(dataset_root, "**", "*.avi"), recursive=True)
    if not video_files:
        video_files = glob.glob(os.path.join(dataset_root, "**", "*.mp4"), recursive=True)

    print(f"🔍 正在扫描 {len(video_files)} 个样本...")

    suspicious_samples = []
    valid_lags = []

    for v_path in tqdm(video_files):
        base = os.path.splitext(v_path)[0]
        a_path = base + ".flac" if os.path.exists(base + ".flac") else base + ".wav"
        if not os.path.exists(a_path): continue

        v, a = extract_signals(v_path, a_path)
        if v is None: continue

        # 计算互相关
        corr = signal.correlate(v, a, mode='full')
        lags = signal.correlation_lags(len(v), len(a), mode='full')

        # 获取最佳匹配点的索引
        peak_idx = np.argmax(corr)
        lag = lags[peak_idx]

        # 计算 "归一化相关系数" (Confidence Score)
        # 范围 [-1, 1], 值越大说明信号形状越匹配
        confidence = corr[peak_idx] / (np.linalg.norm(v) * np.linalg.norm(a))

        # 记录有效数据
        valid_lags.append(lag)

        # 捕获异常：如果 Lag < 0 或者 Lag > 100 (极端大值)
        # 且仅当样本确实存在时
        if lag < 0:
            suspicious_samples.append({
                'file': os.path.basename(v_path),
                'lag': lag,
                'confidence': confidence,
                'v_sig': v,
                'a_sig': a
            })

    print("\n" + "=" * 40)
    print("🕵️‍♀️ 异常分析报告")
    print("=" * 40)
    print(f"负延迟样本数 (Lag < 0): {len(suspicious_samples)}")

    # 按置信度排序，看看是不是低置信度的全是乱七八糟的
    suspicious_samples.sort(key=lambda x: x['confidence'])

    print("\n--- 典型异常样本 (Top 3 低置信度) ---")
    for i in range(min(3, len(suspicious_samples))):
        s = suspicious_samples[i]
        print(f"File: {s['file']}, Lag: {s['lag']}, Conf: {s['confidence']:.4f}")

    # --- 可视化前3个异常样本 ---
    if len(suspicious_samples) > 0:
        plt.figure(figsize=(12, 8))
        for i in range(min(3, len(suspicious_samples))):
            s = suspicious_samples[i]
            plt.subplot(3, 1, i + 1)
            plt.title(f"Negative Lag Analysis: {s['file']} (Lag={s['lag']}, Conf={s['confidence']:.2f})")

            # 画出按照计算出的 Lag 移动后的样子
            # 如果 Lag 是负数 (例如 -100)，意味着视频要向右移 (延迟播放) 才能对上？
            # 不，signal.correlate 的 lag 定义是: b 移动多少能对上 a
            # 这里为了简单展示，我们直接画原始波形
            plt.plot(s['a_sig'], 'b', label='Audio', alpha=0.6)
            plt.plot(s['v_sig'], 'r', label='Video', alpha=0.6)
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("debug_negative_lags.png")
        print("\n📸 已保存异常样本波形图: debug_negative_lags.png")
        print("   -> 请检查图中是否有一条线是平的？或者全是噪音？")

    # --- 重新计算稳健统计量 (Robust Statistics) ---
    lags_array = np.array(valid_lags)

    # 使用四分位距 (IQR) 过滤异常值
    Q1 = np.percentile(lags_array, 25)
    Q3 = np.percentile(lags_array, 75)
    IQR = Q3 - Q1

    # 定义由于离群点导致的 "正常范围"
    # 通常是 1.5 * IQR，但在信号延迟中我们可以宽松一点
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 再次过滤
    clean_lags = lags_array[(lags_array >= lower_bound) & (lags_array <= upper_bound)]

    print("\n" + "=" * 40)
    print("✅ 清洗后的推荐先验 (Robust Statistics)")
    print("=" * 40)
    print(f"原始均值: {np.mean(lags_array):.2f}")
    print(f"原始范围: [{np.min(lags_array)}, {np.max(lags_array)}]")
    print("-" * 20)
    print(f"Q1 (25%): {Q1:.2f}")
    print(f"Q3 (75%): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"清洗阈值: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print("-" * 20)
    print(f"清洗后均值 (Mean): {np.mean(clean_lags):.2f}")
    print(f"清洗后标准差 (Std): {np.std(clean_lags):.2f}")
    print(f"💡 最终建议 CAVA 范围: [{np.floor(np.min(clean_lags))}, {np.ceil(np.max(clean_lags))}]")
    print("=" * 40)


if __name__ == "__main__":
    DATA_PATH = "./intel_robotic_welding_dataset/"
    if os.path.exists(DATA_PATH):
        debug_negative_lags(DATA_PATH)