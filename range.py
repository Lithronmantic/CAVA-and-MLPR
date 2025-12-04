import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import librosa
import os
import glob
from tqdm import tqdm


# ==========================================
# 1. 修改后的 Dataset：只返回原始 Raw 数据，不做任何对齐
# ==========================================
class RawAnalysisDataset(Dataset):
    def __init__(self, root_dir):
        self.file_pairs = self._find_files(root_dir)
        print(f"📂 找到 {len(self.file_pairs)} 对样本用于统计分析")

    def _find_files(self, root_dir):
        pairs = []
        # 递归查找所有视频
        video_files = glob.glob(os.path.join(root_dir, "**", "*.avi"), recursive=True)
        if not video_files:
            video_files = glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True)

        for v_path in video_files:
            base_path = os.path.splitext(v_path)[0]
            # 尝试匹配音频
            for ext in ['.flac', '.wav']:
                a_path = base_path + ext
                if os.path.exists(a_path):
                    pairs.append((v_path, a_path))
                    break
        return pairs

    def _extract_signals(self, video_path, audio_path):
        # --- 视频提取 ---
        cap = cv2.VideoCapture(video_path)
        visual_energy = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            visual_energy.append(np.mean(gray))
        cap.release()
        visual_energy = np.array(visual_energy)

        # 归一化 (Zero-mean 对于互相关非常重要)
        if np.std(visual_energy) > 1e-5:
            visual_energy = (visual_energy - np.mean(visual_energy)) / np.std(visual_energy)

        # --- 音频提取 ---
        y, sr = librosa.load(audio_path, sr=None)
        target_len = len(visual_energy)
        if target_len == 0: return None, None

        # 计算 hop_length 以对齐视频帧数
        samples_per_frame = int(len(y) / target_len)
        if samples_per_frame == 0: return None, None

        audio_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=samples_per_frame, center=True)[0]

        # 强制长度对齐 (截断多余部分)
        min_len = min(len(visual_energy), len(audio_rms))
        visual_energy = visual_energy[:min_len]
        audio_rms = audio_rms[:min_len]

        # 归一化
        if np.std(audio_rms) > 1e-5:
            audio_rms = (audio_rms - np.mean(audio_rms)) / np.std(audio_rms)

        return visual_energy, audio_rms

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, index):
        video_path, audio_path = self.file_pairs[index]
        v, a = self._extract_signals(video_path, audio_path)
        if v is None:
            return None
        return v, a, os.path.basename(video_path)


# ==========================================
# 2. 核心统计函数：互相关分析 (Cross-Correlation)
# ==========================================
def calculate_dataset_delays(dataset_root):
    dataset = RawAnalysisDataset(root_dir=dataset_root)
    delays = []

    print("🚀 开始全量数据时延统计 (Using Cross-Correlation)...")

    # 遍历所有数据
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        if sample is None: continue

        v_sig, a_sig, fname = sample

        # 使用 scipy.signal.correlate 计算互相关
        # mode='full' 会计算所有可能的偏移
        correlation = signal.correlate(v_sig, a_sig, mode='full')
        lags = signal.correlation_lags(len(v_sig), len(a_sig), mode='full')

        # 找到相关性最大的位置
        peak_idx = np.argmax(correlation)
        lag_frames = lags[peak_idx]

        # 注意：lag_frames 表示 v 相对于 a 的位移
        # 如果 lag 是负数，说明视频比音频早；如果是正数，说明视频比音频晚 (Lag)
        # 根据你的描述，视频是 Lag (滞后) 的，所以我们预期这里大部分是正值
        delays.append(lag_frames)

    delays = np.array(delays)

    # --- 统计分析 ---
    mean_lag = np.mean(delays)
    std_lag = np.std(delays)
    min_lag = np.min(delays)
    max_lag = np.max(delays)

    # 使用 3-Sigma 准则确定置信区间 (覆盖 99.7% 的样本)
    # 或者使用 2-Sigma (覆盖 95%)，视数据脏乱程度而定
    # 这里我们保守一点，使用 mean ± 3*std，并结合 min/max

    # 建议的先验范围 (取整)
    suggested_low = np.floor(mean_lag - 3 * std_lag)
    suggested_high = np.ceil(mean_lag + 3 * std_lag)

    print("\n" + "=" * 40)
    print("📊 数据集时延统计结果 (Data-Driven Prior)")
    print("=" * 40)
    print(f"样本总数: {len(delays)}")
    print(f"平均时延 (Mean Lag): {mean_lag:.2f} 帧")
    print(f"标准差 (Std Dev):   {std_lag:.2f} 帧")
    print(f"最小观测值 (Min):   {min_lag} 帧")
    print(f"最大观测值 (Max):   {max_lag} 帧")
    print("-" * 40)
    print(f"💡 建议 CAVA 设置范围 (Mean ± 3σ):")
    print(f"   delta_low_frames  = {suggested_low:.1f}")
    print(f"   delta_high_frames = {suggested_high:.1f}")
    print("=" * 40)

    # --- 可视化分布图 ---
    plt.figure(figsize=(10, 6))
    sns.histplot(delays, bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(mean_lag, color='red', linestyle='--', label=f'Mean: {mean_lag:.2f}')
    plt.axvline(suggested_low, color='green', linestyle=':', label='Lower Bound (-3std)')
    plt.axvline(suggested_high, color='green', linestyle=':', label='Upper Bound (+3std)')

    plt.title(f"Distribution of Audio-Video Latency (Prior Knowledge)\nDataset: {os.path.basename(dataset_root)}")
    plt.xlabel("Lag (Frames) [Positive means Video comes after Audio]")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)

    save_path = "latency_distribution_prior.png"
    plt.savefig(save_path, dpi=300)
    print(f"📈 分布直方图已保存至: {save_path}")
    print("   (请将此图作为论文中的 Prior Knowledge 依据)")


if __name__ == "__main__":
    # 替换为你的数据路径
    DATA_PATH = "./intel_robotic_welding_dataset/"
    if os.path.exists(DATA_PATH):
        calculate_dataset_delays(DATA_PATH)
    else:
        print("❌ 路径错误")