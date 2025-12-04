#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrongEvaluator with Publication-Quality Visualizations
包含CAVA和MLPR的深度特征分析可视化
"""
import os, sys, argparse, yaml, random, numpy as np, torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve,
                             precision_recall_curve, auc)
from sklearn.manifold import TSNE
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')

# 固定唯一来源
from dataset import AVFromCSV, safe_collate_fn

# 尝试导入完整版EnhancedDetector
try:
    # 优先从scripts导入（完整版）
    from scripts.enhanced_detector import EnhancedAVTopDetector

    print("✓ 使用 scripts/enhanced_detector.py (完整版)")
except ImportError:
    try:
        # 降级到根目录版本
        from enhanced_detector import EnhancedAVTopDetector

        print("⚠ 使用根目录 enhanced_detector.py (可能是简化版)")
    except ImportError:
        raise ImportError("无法找到 EnhancedAVTopDetector，请检查模型文件位置")

# 设置matplotlib参数以达到出版质量
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_json_serializable(obj):
    """将对象转换为JSON可序列化格式"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else float(obj.item())
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


class StrongEvaluator:
    """增强版评估器 - 包含顶刊级别可视化"""

    def __init__(self, cfg: dict, checkpoint_path: str, out_dir: str):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 创建可视化子目录
        self.vis_dir = self.out_dir / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _set_seed(int(cfg.get("seed", 42)))

        # 加载checkpoint
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 数据配置
        data_cfg = cfg["data"]
        self.C = int(data_cfg["num_classes"])
        self.num_classes = self.C
        self.class_names = list(data_cfg["class_names"])

        # 构建模型
        model_cfg = dict(cfg.get("model", {}))
        model_cfg["num_classes"] = self.C
        fusion_cfg = model_cfg.get("fusion", cfg.get("fusion", {}))

        print(f"📦 构建模型: EnhancedAVTopDetector")
        print(f"  - num_classes: {self.C}")
        print(f"  - fusion_type: {fusion_cfg.get('type', 'N/A')}")
        print(f"  - CAVA enabled: {cfg.get('cava', {}).get('enabled', False)}")

        self.model = EnhancedAVTopDetector({"model": model_cfg, "fusion": fusion_cfg, "cava": cfg.get("cava", {})})

        # 打印模型结构信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        # 验证模型是否有关键组件
        has_video = hasattr(self.model, 'video_encoder')
        has_audio = hasattr(self.model, 'audio_encoder')
        has_fusion = hasattr(self.model, 'fusion') or hasattr(self.model, 'coattn')
        has_cava = hasattr(self.model, 'cava')
        print(f"  - Video encoder: {'✓' if has_video else '✗'}")
        print(f"  - Audio encoder: {'✓' if has_audio else '✗'}")
        print(f"  - Fusion module: {'✓' if has_fusion else '✗'}")
        print(f"  - CAVA module: {'✓' if has_cava else '✗'}")

        sd = ckpt.get('state_dict', ckpt)
        missing_keys, unexpected_keys = self.model.load_state_dict(sd, strict=False)

        if missing_keys:
            print(f"  ⚠ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  ⚠ Unexpected keys: {len(unexpected_keys)}")

        self.model.to(self.device)
        self.model.eval()
        print(f"✓ 模型加载完成: {self.num_classes} classes")

        # 检测CAVA功能
        self.has_cava = hasattr(self.model, 'cava') and self.model.cava is not None
        print(f"✓ CAVA模块: {'启用' if self.has_cava else '禁用'}")

        # 构建测试数据加载器
        root = data_cfg.get("data_root", "")
        vcfg = cfg.get("video", {})
        acfg = cfg.get("audio", {})
        test_csv = data_cfg.get("test_csv") or data_cfg.get("val_csv")

        if not test_csv or not os.path.exists(test_csv):
            raise FileNotFoundError(f"测试集 CSV 不存在: {test_csv}")

        self.ds_test = AVFromCSV(test_csv, root, self.C, self.class_names, vcfg, acfg, is_unlabeled=False)

        tr = cfg.get("training", {})
        bs = int(tr.get("batch_size", 16))
        pin_mem = (self.device.type == 'cuda')

        self.loader = torch.utils.data.DataLoader(
            self.ds_test,
            batch_size=bs,
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers_val", 4)),
            pin_memory=pin_mem,
            collate_fn=safe_collate_fn
        )

        print(f"✓ 测试集: {len(self.ds_test)} 样本, {len(self.loader)} batches")

    def _unpack_batch(self, b):
        """健壮的batch解包"""
        if isinstance(b, dict):
            keys = {k.lower(): k for k in b.keys()}
            v = b[keys['video']]
            a = b[keys['audio']]
            y = b[keys['label']]
            ids = b.get(keys.get('ids'))
            return v, a, y, ids

        if isinstance(b, (list, tuple)):
            v, a, y = b[:3]
            ids = b[3] if len(b) >= 4 else None
            return v, a, y, ids

        raise ValueError(f"Unsupported batch type: {type(b)}")

    @torch.no_grad()
    def evaluate(self):
        """执行评估 - 主函数"""
        # 收集数据
        all_preds, all_labels, all_probs, all_ids = [], [], [], []
        all_video_feats, all_audio_feats, all_fused_feats = [], [], []
        all_cava_gates, all_cava_delays = [], []
        all_frame_preds, all_attention_weights = [], []

        print("\n" + "=" * 80)
        print("🔍 开始评估（收集详细特征）...")
        print("=" * 80)

        for batch_idx, batch in enumerate(tqdm(self.loader, desc="Evaluating")):
            v, a, y, ids = self._unpack_batch(batch)

            if hasattr(y, "ndim") and y.ndim == 2:
                y = y.argmax(dim=1)

            v, a, y = v.to(self.device), a.to(self.device), y.to(self.device)

            # 前向
            out = self.model(v, a)

            logits = out["clip_logits"] if isinstance(out, dict) else out
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if ids is not None:
                all_ids.extend(ids)

            # 收集特征（限制数量）
            if batch_idx < 100 and isinstance(out, dict):
                if "video_proj" in out and out["video_proj"] is not None:
                    all_video_feats.append(out["video_proj"].cpu().numpy())
                if "audio_aligned" in out and out["audio_aligned"] is not None:
                    all_audio_feats.append(out["audio_aligned"].cpu().numpy())
                if "fused_feat" in out and out["fused_feat"] is not None:
                    all_fused_feats.append(out["fused_feat"].cpu().numpy())
                if self.has_cava:
                    if "causal_gate" in out and out["causal_gate"] is not None:
                        all_cava_gates.append(out["causal_gate"].cpu().numpy())
                    if "pred_delay" in out and out["pred_delay"] is not None:
                        all_cava_delays.append(out["pred_delay"].cpu().numpy())
                if "frame_logits" in out and out["frame_logits"] is not None:
                    frame_probs = torch.softmax(out["frame_logits"], dim=-1)
                    all_frame_preds.append(frame_probs.cpu().numpy())
                if "attention_weights" in out and out["attention_weights"] is not None:
                    all_attention_weights.append(out["attention_weights"].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        results = self._compute_metrics(all_labels, all_preds, all_probs)
        self._save_results(results, all_labels, all_preds, all_probs, all_ids)

        # 生成可视化
        print("\n" + "=" * 80)
        print("📊 生成顶刊级别可视化...")
        print("=" * 80)

        self._visualize_confusion_matrix(all_labels, all_preds)
        self._visualize_roc_pr_curves(all_labels, all_probs)
        self._visualize_per_class_performance(all_labels, all_preds)
        self._visualize_confidence_distribution(all_labels, all_probs, all_preds)

        if len(all_cava_gates) > 0:
            self._visualize_cava_analysis(all_cava_gates, all_cava_delays, all_labels, all_preds)

        if len(all_fused_feats) > 0:
            self._visualize_feature_space(all_video_feats, all_audio_feats, all_fused_feats,
                                          all_labels[:len(all_fused_feats) * self.loader.batch_size])

        if len(all_frame_preds) > 0:
            self._visualize_frame_predictions(all_frame_preds, all_labels, all_preds)

        if len(all_attention_weights) > 0:
            self._visualize_attention_weights(all_attention_weights)

        self._visualize_error_analysis(all_labels, all_preds, all_probs, all_ids)

        print(f"\n✓ 所有可视化已保存至: {self.vis_dir}")

        return results

    def _compute_metrics(self, labels, preds, probs):
        """计算评估指标"""
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        precision_per_cls, recall_per_cls, f1_per_cls, support = precision_recall_fscore_support(labels, preds,
                                                                                                 average=None,
                                                                                                 zero_division=0)

        auc_score = None
        if self.num_classes == 2:
            try:
                auc_score = roc_auc_score(labels, probs[:, 1])
            except:
                pass

        cm = confusion_matrix(labels, preds)

        results = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': cm.tolist(),
            'per_class': {
                'precision': precision_per_cls.tolist(),
                'recall': recall_per_cls.tolist(),
                'f1': f1_per_cls.tolist(),
                'support': support.tolist()
            },
            'num_samples': len(labels),
            'num_classes': self.num_classes
        }

        print("\n" + "=" * 80)
        print("📊 评估结果")
        print("=" * 80)
        print(f"准确率 (Accuracy):  {acc * 100:.2f}%")
        print(f"精度 (Precision):   {precision * 100:.2f}%")
        print(f"召回率 (Recall):    {recall * 100:.2f}%")
        print(f"F1-Score:           {f1 * 100:.2f}%")
        if auc_score is not None:
            print(f"AUC:                {auc_score:.4f}")
        print(f"\n混淆矩阵:\n{cm}")
        print("=" * 80)

        return results

    def _save_results(self, results, labels, preds, probs, ids):
        """保存结果"""
        # 确保results完全可序列化
        serializable_results = _to_json_serializable(results)

        with open(self.out_dir / "eval_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        np.savez(self.out_dir / "predictions.npz",
                 labels=labels, preds=preds, probs=probs,
                 ids=np.array(ids) if ids else None)

        print(f"\n✓ 结果已保存至: {self.out_dir}")

    def _visualize_confusion_matrix(self, labels, preds):
        """可视化1: 混淆矩阵"""
        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'}, ax=axes[0], linewidths=0.5, square=True, vmin=0)
        axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold', pad=20)
        axes[0].set_xlabel('Predicted Label', fontweight='bold')
        axes[0].set_ylabel('True Label', fontweight='bold')

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': 'Percentage'}, ax=axes[1], linewidths=0.5, square=True, vmin=0, vmax=1)
        axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold', pad=20)
        axes[1].set_xlabel('Predicted Label', fontweight='bold')
        axes[1].set_ylabel('True Label', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.vis_dir / '01_confusion_matrix.png', dpi=300)
        plt.close()
        print(f"  ✓ 混淆矩阵")

    def _visualize_roc_pr_curves(self, labels, probs):
        """可视化2: ROC和PR曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = auc(fpr, tpr)

            axes[0].plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {roc_auc:.3f})')
            axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            axes[0].set_xlabel('False Positive Rate', fontweight='bold')
            axes[0].set_ylabel('True Positive Rate', fontweight='bold')
            axes[0].set_title('ROC Curve', fontweight='bold', pad=20)
            axes[0].legend(loc="lower right")
            axes[0].grid(True, alpha=0.3)

            precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
            pr_auc = auc(recall, precision)

            axes[1].plot(recall, precision, color='navy', lw=3, label=f'PR (AUC = {pr_auc:.3f})')
            axes[1].set_xlabel('Recall', fontweight='bold')
            axes[1].set_ylabel('Precision', fontweight='bold')
            axes[1].set_title('Precision-Recall Curve', fontweight='bold', pad=20)
            axes[1].legend(loc="lower left")
            axes[1].grid(True, alpha=0.3)
        else:
            from sklearn.preprocessing import label_binarize
            labels_bin = label_binarize(labels, classes=range(self.num_classes))
            colors = plt.cm.rainbow(np.linspace(0, 1, self.num_classes))

            for i, color in enumerate(colors):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                axes[0].plot(fpr, tpr, color=color, lw=2, label=f'{self.class_names[i]} ({roc_auc:.2f})')

            axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
            axes[0].set_xlabel('False Positive Rate', fontweight='bold')
            axes[0].set_ylabel('True Positive Rate', fontweight='bold')
            axes[0].set_title('Multi-class ROC', fontweight='bold', pad=20)
            axes[0].legend(loc="lower right", fontsize=9)

            for i, color in enumerate(colors):
                precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
                pr_auc = auc(recall, precision)
                axes[1].plot(recall, precision, color=color, lw=2, label=f'{self.class_names[i]} ({pr_auc:.2f})')

            axes[1].set_xlabel('Recall', fontweight='bold')
            axes[1].set_ylabel('Precision', fontweight='bold')
            axes[1].set_title('Multi-class PR', fontweight='bold', pad=20)
            axes[1].legend(loc="lower left", fontsize=9)

        plt.tight_layout()
        plt.savefig(self.vis_dir / '02_roc_pr_curves.png', dpi=300)
        plt.close()
        print(f"  ✓ ROC/PR曲线")

    def _visualize_per_class_performance(self, labels, preds):
        """可视化3: Per-class性能"""
        precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

        accuracy_per_class = []
        for i in range(self.num_classes):
            mask = labels == i
            acc = (preds[mask] == i).sum() / mask.sum() if mask.sum() > 0 else 0
            accuracy_per_class.append(acc)
        accuracy_per_class = np.array(accuracy_per_class)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        x = np.arange(len(self.class_names))
        width = 0.6

        metrics = [
            (accuracy_per_class, 'Accuracy', 'steelblue', axes[0, 0]),
            (precision, 'Precision', 'coral', axes[0, 1]),
            (recall, 'Recall', 'mediumseagreen', axes[1, 0]),
            (f1, 'F1-Score', 'mediumpurple', axes[1, 1])
        ]

        for data, title, color, ax in metrics:
            bars = ax.bar(x, data, width, color=color, alpha=0.8)
            ax.set_ylabel(title, fontweight='bold')
            ax.set_title(f'Per-Class {title}', fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_ylim([0, 1])
            ax.axhline(y=data.mean(), color='r', linestyle='--', lw=2, label=f'Mean={data.mean():.3f}')
            ax.legend()
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2%}', ha='center', va='bottom',
                        fontsize=10)

        plt.tight_layout()
        plt.savefig(self.vis_dir / '03_per_class_performance.png', dpi=300)
        plt.close()
        print(f"  ✓ Per-class性能")

    def _visualize_confidence_distribution(self, labels, probs, preds):
        """可视化4: 置信度分布"""
        correct_mask = (preds == labels)
        correct_conf = probs[np.arange(len(labels)), labels][correct_mask]
        wrong_conf = probs[np.arange(len(labels)), preds][~correct_mask]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].hist(correct_conf, bins=50, alpha=0.7, color='green', label=f'Correct (n={len(correct_conf)})',
                     edgecolor='black')
        axes[0].hist(wrong_conf, bins=50, alpha=0.7, color='red', label=f'Wrong (n={len(wrong_conf)})',
                     edgecolor='black')
        axes[0].axvline(correct_conf.mean(), color='darkgreen', linestyle='--', lw=2.5,
                        label=f'Mean Correct={correct_conf.mean():.3f}')
        axes[0].axvline(wrong_conf.mean(), color='darkred', linestyle='--', lw=2.5,
                        label=f'Mean Wrong={wrong_conf.mean():.3f}')
        axes[0].set_xlabel('Confidence Score', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Confidence Distribution', fontweight='bold', pad=15)
        axes[0].legend()

        bp = axes[1].boxplot([correct_conf, wrong_conf], labels=['Correct', 'Wrong'], patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[1].set_ylabel('Confidence Score', fontweight='bold')
        axes[1].set_title('Confidence Box Plot', fontweight='bold', pad=15)

        plt.tight_layout()
        plt.savefig(self.vis_dir / '04_confidence_distribution.png', dpi=300)
        plt.close()
        print(f"  ✓ 置信度分布")

    def _visualize_cava_analysis(self, gates, delays, labels, preds):
        """可视化5: CAVA分析（核心创新）"""
        gates = np.concatenate(gates, axis=0)
        if len(delays) > 0:
            delays = np.concatenate(delays, axis=0)

        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 门控分布
        ax1 = fig.add_subplot(gs[0, 0])
        # 确保gate_mean是1维数组
        if gates.ndim > 1:
            gate_mean = gates.reshape(len(gates), -1).mean(axis=1)
        else:
            gate_mean = gates

        ax1.hist(gate_mean, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(gate_mean.mean(), color='red', linestyle='--', lw=2.5, label=f'Mean={gate_mean.mean():.3f}')
        ax1.axvline(np.median(gate_mean), color='orange', linestyle='--', lw=2.5,
                    label=f'Median={np.median(gate_mean):.3f}')
        ax1.set_xlabel('Causal Gate Value', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('CAVA Gate Distribution', fontweight='bold', pad=15)
        ax1.legend()

        # 2. 正确vs错误
        ax2 = fig.add_subplot(gs[0, 1])
        correct_mask = (preds[:len(gate_mean)] == labels[:len(gate_mean)])
        bp = ax2.boxplot([gate_mean[correct_mask], gate_mean[~correct_mask]], labels=['Correct', 'Wrong'],
                         patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('Mean Gate Value', fontweight='bold')
        ax2.set_title('CAVA Gate: Correct vs Wrong', fontweight='bold', pad=15)

        # 3. 时间序列
        ax3 = fig.add_subplot(gs[1, 0])
        if gates.ndim >= 2 and gates.shape[1] > 1:
            # 确保是2维 [N, T]
            if gates.ndim > 2:
                gates_2d = gates.reshape(gates.shape[0], -1)
            else:
                gates_2d = gates

            n_samples = min(10, gates_2d.shape[0])
            colors = plt.cm.rainbow(np.linspace(0, 1, n_samples))
            for i in range(n_samples):
                ax3.plot(gates_2d[i], color=colors[i], alpha=0.7, lw=2, label=f'Sample {i + 1}')
            ax3.axhline(y=0.5, color='black', linestyle='--', lw=2, label='Threshold')
            ax3.set_xlabel('Frame', fontweight='bold')
            ax3.set_ylabel('Gate Value', fontweight='bold')
            ax3.set_title('Gate Evolution', fontweight='bold', pad=15)
            ax3.legend(fontsize=8, ncol=2)
            ax3.set_ylim([0, 1])
        else:
            ax3.text(0.5, 0.5, 'No temporal data', ha='center', va='center',
                     fontsize=14, transform=ax3.transAxes)

        # 4. 延迟分布
        ax4 = fig.add_subplot(gs[1, 1])
        if len(delays) > 0 and delays.size > 0:
            try:
                # 确保delays是1维并且是整数
                if delays.ndim > 1:
                    delays_1d = delays.flatten()
                else:
                    delays_1d = delays
                delay_counts = np.bincount(delays_1d.astype(int).clip(0, 20))
                ax4.bar(range(len(delay_counts)), delay_counts, color='mediumpurple', alpha=0.8, edgecolor='black')
                ax4.set_xlabel('Predicted Delay (frames)', fontweight='bold')
                ax4.set_ylabel('Count', fontweight='bold')
                ax4.set_title('Predicted AV Delay', fontweight='bold', pad=15)
                ax4.grid(True, alpha=0.3, axis='y')
            except Exception as e:
                ax4.text(0.5, 0.5, f'Delay data error:\n{str(e)[:30]}', ha='center', va='center',
                         fontsize=12, transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'No delay data', ha='center', va='center',
                     fontsize=14, transform=ax4.transAxes)
            ax4.set_title('Predicted AV Delay', fontweight='bold', pad=15)

        plt.suptitle('CAVA (Causal Audio-Visual Alignment) Analysis', fontsize=20, fontweight='bold')
        plt.savefig(self.vis_dir / '05_cava_analysis.png', dpi=300)
        plt.close()
        print(f"  ✓ CAVA分析")

    def _visualize_feature_space(self, video_feats, audio_feats, fused_feats, labels):
        """可视化6: 特征空间t-SNE"""
        if len(fused_feats) == 0:
            print("  ⚠ 跳过特征空间")
            return

        fused = np.concatenate(fused_feats, axis=0)
        if fused.ndim == 3:
            fused = fused.mean(axis=1)

        n_samples = min(2000, len(fused))
        indices = np.random.choice(len(fused), n_samples, replace=False)
        fused_sample = fused[indices]
        labels_sample = labels[indices]

        print("  → t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        fused_2d = tsne.fit_transform(fused_sample)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_classes))
        for i, color in enumerate(colors):
            mask = labels_sample == i
            ax.scatter(fused_2d[mask, 0], fused_2d[mask, 1], c=[color], label=self.class_names[i], alpha=0.6, s=50,
                       edgecolors='black', linewidth=0.5)

        ax.set_xlabel('t-SNE Dim 1', fontweight='bold')
        ax.set_ylabel('t-SNE Dim 2', fontweight='bold')
        ax.set_title('Feature Space (t-SNE)', fontweight='bold', fontsize=18, pad=20)
        ax.legend(fontsize=11, markerscale=1.5)

        plt.savefig(self.vis_dir / '06_feature_space_tsne.png', dpi=300)
        plt.close()
        print(f"  ✓ 特征空间t-SNE")

    def _visualize_frame_predictions(self, frame_preds, labels, clip_preds):
        """可视化7: 帧级预测"""
        frame_preds = np.concatenate(frame_preds, axis=0)
        n_samples = min(8, len(frame_preds))

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i in range(n_samples):
            probs = frame_preds[i]
            for c in range(self.num_classes):
                axes[i].plot(probs[:, c], label=self.class_names[c], lw=2, alpha=0.8)
            axes[i].axhline(y=0.5, color='gray', linestyle='--', lw=1)
            axes[i].set_xlabel('Frame', fontweight='bold')
            axes[i].set_ylabel('Probability', fontweight='bold')
            axes[i].set_title(f'True={self.class_names[labels[i]]}, Pred={self.class_names[clip_preds[i]]}',
                              fontweight='bold', fontsize=11)
            axes[i].legend(fontsize=8)
            axes[i].set_ylim([0, 1])

        plt.suptitle('Frame-Level Predictions', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.vis_dir / '07_frame_predictions.png', dpi=300)
        plt.close()
        print(f"  ✓ 帧级预测")

    def _visualize_attention_weights(self, attention_weights):
        """可视化8: 注意力权重"""
        attn = np.concatenate(attention_weights, axis=0)

        # 确保是2D矩阵用于绘制
        if attn.ndim > 2:
            # 如果是[N, H, T, T]或其他高维，取均值降维
            while attn.ndim > 2:
                attn = attn.mean(axis=1)

        attn_mean = attn.mean(axis=0) if attn.ndim == 3 else attn.mean(axis=0) if len(attn.shape) > 2 else attn[
            0] if attn.ndim == 2 else attn

        # 确保attn_mean是2维的
        if attn_mean.ndim != 2:
            print(f"  ⚠ 跳过注意力可视化（维度异常: {attn_mean.shape}）")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        im1 = axes[0].imshow(attn_mean, cmap='viridis', aspect='auto')
        axes[0].set_xlabel('Key', fontweight='bold')
        axes[0].set_ylabel('Query', fontweight='bold')
        axes[0].set_title('Average Attention', fontweight='bold', pad=15)
        plt.colorbar(im1, ax=axes[0])

        # 绘制单个样本
        sample_attn = attn[0] if attn.ndim >= 2 else attn_mean
        if sample_attn.ndim > 2:
            sample_attn = sample_attn.mean(axis=0)

        im2 = axes[1].imshow(sample_attn, cmap='plasma', aspect='auto')
        axes[1].set_xlabel('Key', fontweight='bold')
        axes[1].set_ylabel('Query', fontweight='bold')
        axes[1].set_title('Sample Attention', fontweight='bold', pad=15)
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(self.vis_dir / '08_attention_weights.png', dpi=300)
        plt.close()
        print(f"  ✓ 注意力权重")

    def _visualize_error_analysis(self, labels, preds, probs, ids):
        """可视化9: 错误分析"""
        errors = np.where(preds != labels)[0]
        if len(errors) == 0:
            print("  ✓ 无错误样本")
            return

        error_conf = probs[errors, preds[errors]]
        sorted_idx = np.argsort(error_conf)[::-1]
        top_errors = errors[sorted_idx[:min(20, len(errors))]]

        confusion_pairs = {}
        for idx in errors:
            pair = (labels[idx], preds[idx])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        top_n = min(10, len(sorted_pairs))
        pair_labels = [f'{self.class_names[p[0][0]]} → {self.class_names[p[0][1]]}' for p in sorted_pairs[:top_n]]
        pair_counts = [p[1] for p in sorted_pairs[:top_n]]

        y_pos = np.arange(len(pair_labels))
        axes[0].barh(y_pos, pair_counts, color='salmon', alpha=0.8, edgecolor='black')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(pair_labels)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Count', fontweight='bold')
        axes[0].set_title('Top Confusion Pairs', fontweight='bold', pad=15)

        axes[1].hist(error_conf, bins=30, color='coral', alpha=0.7, edgecolor='black')
        axes[1].axvline(error_conf.mean(), color='red', linestyle='--', lw=2.5, label=f'Mean={error_conf.mean():.3f}')
        axes[1].set_xlabel('Error Confidence', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Error Confidence', fontweight='bold', pad=15)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.vis_dir / '09_error_analysis.png', dpi=300)
        plt.close()
        print(f"  ✓ 错误分析")

        # 保存错误列表
        error_report = []
        for idx in top_errors:
            idx = int(idx)  # 确保索引是Python int
            true_idx = int(labels[idx])
            pred_idx = int(preds[idx])

            error_dict = {
                'id': idx if not ids else _to_json_serializable(ids[idx]),
                'true_label': self.class_names[true_idx],
                'pred_label': self.class_names[pred_idx],
                'confidence': float(probs[idx, pred_idx]),
                'true_prob': float(probs[idx, true_idx])
            }
            error_report.append(error_dict)

        with open(self.out_dir / 'error_cases.json', 'w') as f:
            json.dump(error_report, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='🔍 StrongEvaluator with Publication Visualizations')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/eval")
    parser.add_argument("--model-path", type=str, default=None,
                        help="自定义模型路径，例如: scripts.enhanced_detector")
    args = parser.parse_args()

    # 如果指定了自定义模型路径，动态导入
    if args.model_path:
        print(f"📦 使用自定义模型路径: {args.model_path}")
        import importlib
        module = importlib.import_module(args.model_path)
        global EnhancedAVTopDetector
        EnhancedAVTopDetector = module.EnhancedAVTopDetector

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 80)
    print("🔍 Enhanced Evaluation with CAVA & Feature Analysis")
    print("=" * 80)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.output}")
    print("=" * 80)

    evaluator = StrongEvaluator(cfg, args.checkpoint, args.output)
    evaluator.evaluate()

    print("\n" + "=" * 80)
    print("✅ 评估完成! 已生成顶刊级别可视化")
    print("=" * 80)
    print(f"📂 结果: {args.output}")
    print(f"📊 可视化: {args.output}/visualizations/")


if __name__ == "__main__":
    main()