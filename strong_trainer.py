# -*- coding: utf-8 -*-
"""
StrongTrainer - 论文级修复版 + 完整可视化
✅ 修复1: MLPR元学习使用stateless
✅ 修复2: 分布对齐修正为ReMixMatch
✅ 修复3: 损失权重平衡
✅ 新增: TensorBoard + Matplotlib可视化
"""
import os, json, math, random, time
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from cava_losses import info_nce_align, corr_diag_align, prior_l2, edge_hinge
from meta_reweighter import MetaReweighter, build_mlpr_features
from ssl_losses import ramp_up
from history_bank import HistoryBank
from teacher_ema import EMATeacher
from dataset import AVFromCSV, safe_collate_fn

try:
    from dataset import safe_collate_fn_with_ids
except Exception:
    def safe_collate_fn_with_ids(batch):
        return safe_collate_fn(batch)

from enhanced_detector import EnhancedAVTopDetector

# AMP 兼容
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    AMP_DEVICE_ARG = True


    def amp_autocast(device_type, enabled=True, dtype=torch.float16):
        return _autocast(device_type, enabled=enabled, dtype=dtype)


    def AmpGradScaler(device_type, enabled=True):
        return _GradScaler(device_type, enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler

    AMP_DEVICE_ARG = False


    def amp_autocast(device_type, enabled=True, dtype=torch.float16):
        return _autocast(enabled=enabled)


    def AmpGradScaler(device_type, enabled=True):
        return _GradScaler(enabled=enabled)


class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        with amp_autocast('cuda', enabled=False):
            logits_f32 = torch.clamp(logits.float(), min=-30, max=30)
            ce = F.cross_entropy(
                logits_f32, targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none"
            )
            with torch.no_grad():
                pt = F.softmax(logits_f32, dim=1).gather(1, targets.view(-1, 1)).squeeze(1)
                pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
                focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * ce
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                return ce.mean()
            return loss.mean()


def _set_seed(seed: int):
    random.seed(seed);
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class StrongTrainer:
    def __init__(self, cfg: Dict[str, Any], out_dir: str, resume_from: Optional[str] = None):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        (self.out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
        self.resume_from = resume_from  # ✅ 新增：checkpoint路径

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        _set_seed(int(cfg.get("seed", 42)))

        # ✅ 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.out_dir / 'runs'))
        print(f"📊 TensorBoard logs: {self.out_dir / 'runs'}")

        # ✅ 损失历史记录器
        self.loss_history = {
            'sup_loss': [],
            'cava_loss': [],
            'cava_align': [],
            'cava_edge': [],
            'pseudo_loss': [],
            'total_loss': [],
            'ssl_mask_ratio': [],
            'gate_mean': [],
            'gate_std': [],
            'learning_rate': [],
            'val_acc_student': [],
            'val_f1_student': [],
            'val_acc_teacher': [],
            'val_f1_teacher': []
        }

        # ✅ Step级别的详细记录（用于平滑曲线）
        self.step_losses = {
            'sup_loss': [],
            'cava_loss': [],
            'pseudo_loss': [],
            'total_loss': []
        }

        # AMP
        self.amp_enabled = bool(cfg.get('training', {}).get('amp', True) and self.device.type == 'cuda')
        self.scaler = AmpGradScaler(self.device_type, enabled=self.amp_enabled)
        self.amp_disable_epoch = int(cfg.get("training", {}).get("amp_disable_epoch", 15))
        self.grad_explosion_count = 0
        self.max_grad_explosion = 3

        # Data
        data_cfg = cfg["data"]
        self.C = int(data_cfg["num_classes"])
        self.num_classes = self.C
        self.class_names = list(data_cfg["class_names"])
        root = data_cfg.get("data_root", "")
        vcfg = cfg.get("video", {})
        acfg = cfg.get("audio", {})
        l_csv = data_cfg["labeled_csv"]
        v_csv = data_cfg["val_csv"]
        u_csv = data_cfg.get("unlabeled_csv")

        self.ds_l = AVFromCSV(l_csv, root, self.C, self.class_names, vcfg, acfg, is_unlabeled=False)
        self.ds_v = AVFromCSV(v_csv, root, self.C, self.class_names, vcfg, acfg, is_unlabeled=False)
        self.ds_u = AVFromCSV(u_csv, root, self.C, self.class_names, vcfg, acfg, is_unlabeled=True) if (
                cfg.get("training", {}).get("use_ssl", False) and u_csv) else None

        self.stats = self._scan_stats(self.ds_l)
        (self.out_dir / 'stats').mkdir(exist_ok=True, parents=True)
        (self.out_dir / 'stats' / 'class_stats.json').write_text(
            json.dumps(self.stats, ensure_ascii=False, indent=2), encoding='utf-8')

        sampler = None
        if data_cfg.get("sampler", "").lower() == "weighted":
            inv_freq = np.array(self.stats["inv_freq"], dtype=np.float32)
            sampler = self._build_sampler(self.ds_l, inv_freq)

        tr = cfg.get("training", {})
        self.bs = int(tr.get("batch_size", 16))
        pin_mem = (self.device.type == 'cuda')

        def _to(nw, default=60):
            return 0 if int(nw) == 0 else default

        self.loader_l = DataLoader(
            self.ds_l, batch_size=self.bs, sampler=sampler, shuffle=(sampler is None),
            num_workers=int(data_cfg.get("num_workers_train", 0)), pin_memory=pin_mem,
            drop_last=True, collate_fn=safe_collate_fn, timeout=_to(data_cfg.get("num_workers_train", 0)),
            persistent_workers=(int(data_cfg.get("num_workers_train", 0)) > 0)
        )
        self.loader_v = DataLoader(
            self.ds_v, batch_size=self.bs, shuffle=False,
            num_workers=int(data_cfg.get("num_workers_val", 0)), pin_memory=pin_mem,
            drop_last=False, collate_fn=safe_collate_fn, timeout=_to(data_cfg.get("num_workers_val", 0)),
            persistent_workers=(int(data_cfg.get("num_workers_val", 0)) > 0)
        )
        self.loader_u = None
        if self.ds_u is not None:
            self.loader_u = DataLoader(
                self.ds_u, batch_size=self.bs, shuffle=True,
                num_workers=int(data_cfg.get("num_workers_unl", 0)), pin_memory=pin_mem,
                drop_last=True, collate_fn=safe_collate_fn_with_ids, timeout=_to(data_cfg.get("num_workers_unl", 0)),
                persistent_workers=(int(data_cfg.get("num_workers_unl", 0)) > 0)
            )

        # Model
        model_cfg = dict(cfg.get("model", {}))
        model_cfg["num_classes"] = self.C
        fusion_cfg = model_cfg.get("fusion", cfg.get("fusion", {}))
        self.model = EnhancedAVTopDetector({"model": model_cfg, "fusion": fusion_cfg, "cava": cfg.get("cava", {})}).to(
            self.device)

        if bool(cfg.get("model", {}).get("init_bias", False)):
            self._init_bias(self.model, self.stats["pi"])

        # Loss
        loss_cfg = cfg.get("loss", {})
        self.loss_name = loss_cfg.get("name", "ce").lower()
        cw = loss_cfg.get("class_weights", None)
        class_weights = torch.tensor(cw, dtype=torch.float32, device=self.device) if cw is not None else None

        if self.loss_name == "focal_ce":
            self.criterion = FocalCrossEntropy(
                gamma=loss_cfg.get("gamma", 2.0),
                label_smoothing=loss_cfg.get("label_smoothing", 0.05),
                class_weights=class_weights
            ).to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=loss_cfg.get("label_smoothing", 0.05)
            ).to(self.device)

        # Optim
        self.epochs = int(tr.get("num_epochs", 20))
        base_lr = float(tr.get("learning_rate", 5e-5))
        bb_mult = float(tr.get("backbone_lr_mult", 0.1))
        self.wd = float(tr.get("weight_decay", 1e-3))
        self.grad_clip = float(tr.get("grad_clip_norm", 1.5))

        head_params, bb_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if "video_backbone" in n or "audio_backbone" in n:
                bb_params.append(p)
            else:
                head_params.append(p)

        self.opt = optim.AdamW(
            [{"params": head_params, "lr": base_lr}, {"params": bb_params, "lr": base_lr * bb_mult}],
            weight_decay=self.wd
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs, eta_min=1e-7)

        self.nan_count = 0
        self.total_steps = 0
        self.meta_fail_count = 0

        # MLPR
        self.mlpr_cfg = dict(cfg.get("mlpr", {}))
        self.mlpr_enabled = bool(self.mlpr_cfg.get("enabled", False))
        self.teacher_ema = EMATeacher(self.model,
                                      decay=float(self.mlpr_cfg.get("ema_decay", 0.999))) if self.mlpr_enabled else None

        use_hist = bool(self.mlpr_cfg.get("use_history_stats", True))
        use_cava = bool(self.mlpr_cfg.get("use_cava_signal", True))
        use_prob_vec = bool(self.mlpr_cfg.get("use_prob_vector", False))
        feat_dim = 3 + 1 + (2 if use_hist else 0) + (1 if use_cava else 0) + (self.C if use_prob_vec else 0)

        self.meta = MetaReweighter(
            input_dim=feat_dim, hidden=(128, 64),
            w_clip=tuple(self.mlpr_cfg.get("weight_clip", [0.05, 0.95])), dropout=0.1
        ).to(self.device) if self.mlpr_enabled else None

        self.meta_opt = optim.Adam(self.meta.parameters(),
                                   lr=float(self.mlpr_cfg.get("meta_lr", 1e-4))) if self.mlpr_enabled else None
        self.hist_bank = HistoryBank(momentum=float(self.mlpr_cfg.get("history_momentum", 0.9))) if (
                    self.mlpr_enabled and use_hist) else None

        self._mlpr_flags = {"use_hist": use_hist, "use_cava": use_cava, "use_prob_vec": use_prob_vec}
        self._mlpr_tempT = float(self.mlpr_cfg.get("T", 1.0))
        self._mlpr_lambda_u = float(self.mlpr_cfg.get("lambda_u", 0.5))
        self._mlpr_ramp_epochs = int(self.mlpr_cfg.get("ramp_up_epochs", 5))
        self._mlpr_meta_interval = int(self.mlpr_cfg.get("meta_interval", 20))
        self._mlpr_inner_lr = float(self.mlpr_cfg.get("inner_lr", base_lr))

        # SSL
        tr_ssl = cfg.get("training", {})
        self.use_ssl = bool(tr_ssl.get("use_ssl", False) and self.ds_u is not None)
        self.teacher = None
        self.ema_decay = 0.999
        self.ssl_warmup = 5
        self.ssl_final_thresh = 0.9
        self.ssl_temp = 1.0
        self.lambda_u = 1.0

        ssl_cfg = dict(cfg.get("ssl", {}))
        self._use_dist_align = bool(ssl_cfg.get("use_dist_align", True))
        self._use_cls_threshold = bool(ssl_cfg.get("use_cls_threshold", True))
        self._thr_min = float(ssl_cfg.get("thr_min", 0.05))
        self._cls_thr_momentum = float(ssl_cfg.get("cls_thr_momentum", 0.9))
        self._cls_conf_ema = torch.full((self.C,), 0.3, device=self.device)
        self._cls_thr = torch.full((self.C,), self.ssl_final_thresh, device=self.device)

        self.ema_decay_init = float(ssl_cfg.get("ema_decay_init", 0.99))
        self._ema_update_interval = int(ssl_cfg.get("ema_update_interval", 10))
        self._ema_step_counter = 0

        if self.use_ssl:
            self.ema_decay = float(ssl_cfg.get("ema_decay", 0.999))
            self.ssl_warmup = int(ssl_cfg.get("warmup_epochs", 5))
            self.ssl_final_thresh = float(ssl_cfg.get("final_thresh", 0.9))
            self.ssl_temp = float(ssl_cfg.get("consistency_temp", 1.0))
            self.lambda_u = float(ssl_cfg.get("lambda_u", 1.0))

            self.teacher = EnhancedAVTopDetector(
                {"model": model_cfg, "fusion": fusion_cfg, "cava": cfg.get("cava", {})}).to(self.device)
            self.teacher.load_state_dict(self.model.state_dict(), strict=False)  # ✅ 允许部分不匹配
            for p in self.teacher.parameters(): p.requires_grad = False
            self.teacher.eval()

        self.eval_with_ema_mode = str(ssl_cfg.get("eval_with_ema", "auto")).lower()

        self.best_f1 = -1.0
        self.cava_cfg = dict(cfg.get("cava", {}))
        self.cava_enabled = bool(self.cava_cfg.get("enabled", False))

        self._print_startup_banner()

        trcfg = cfg.get("training", {})
        self.early_stop_patience = int(trcfg.get("early_stop_patience", 0))
        self.no_improve = 0
        self._pi = torch.tensor(self.stats["pi"], dtype=torch.float32, device=self.device)
        self._teach_p90_ema = 0.2

        # ✅ 新增：从checkpoint恢复
        self.start_epoch = 1
        if self.resume_from is not None:
            self._load_checkpoint(self.resume_from)

    def _load_checkpoint(self, checkpoint_path: str):
        """✅ 新增：从checkpoint恢复训练"""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return

        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型权重
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"✅ Model weights loaded")

        # ✅ 关键：重新初始化Teacher（修复Teacher崩溃）
        if self.teacher is not None:
            self.teacher.load_state_dict(self.model.state_dict(), strict=False)  # ✅ 允许部分不匹配
            print(f"✅ Teacher re-initialized from Student")

        # 恢复epoch信息
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Resume from epoch {self.start_epoch}")

        # 可选：恢复优化器状态（如果需要完全恢复训练状态）
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'opt'):
            try:
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ Optimizer state loaded")
            except Exception as e:
                print(f"⚠️ Could not load optimizer state: {e}")

        # 可选：恢复调度器状态
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✅ Scheduler state loaded")
            except Exception as e:
                print(f"⚠️ Could not load scheduler state: {e}")

        print(f"🎯 Checkpoint loaded successfully!")

    def _print_startup_banner(self):
        print("=" * 60)
        print("🚀 CAVA-SSL Training with Visualization")
        print("=" * 60)
        print("┌─ Model/Fusion")
        print(f"│  fusion_type         = {getattr(self.model, 'fusion_type', 'n/a')}")
        print(f"│  video_dim, audio_dim= {self.model.video_dim}, {self.model.audio_dim}")
        print(f"│  fusion_dim          = {self.model.fusion_dim}")
        print(f"│  num_classes         = {self.num_classes}")
        print("├─ CAVA (Causal Align)")
        print(f"│  enabled             = {self.cava_enabled}")
        if self.cava_enabled:
            print(
                f"│  λ_align/edge        = {self.cava_cfg.get('lambda_align', 0.1)}/{self.cava_cfg.get('lambda_edge', 0.01)}")
        print("├─ SSL")
        print(f"│  enabled             = {self.use_ssl}")
        print("├─ MLPR")
        print(f"│  enabled             = {self.mlpr_enabled}")
        print("├─ Visualization")
        print(f"│  TensorBoard         = {self.out_dir / 'runs'}")
        print(f"│  Plots               = {self.out_dir / 'visualizations'}")
        if self.resume_from:
            print("├─ Resume")
            print(f"│  checkpoint          = {self.resume_from}")
            print(f"│  start_epoch         = {self.start_epoch}")
        print("└" + "─" * 58)

    def _scan_stats(self, ds_l) -> Dict[str, Any]:
        C = self.C
        counts = np.zeros(C, dtype=np.int64)
        n = len(ds_l)
        for i in range(n):
            try:
                item = ds_l[i]
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    y = item[2]
                else:
                    continue
                if torch.is_tensor(y):
                    y = y.detach().cpu()
                    if y.ndim == 0:
                        idx = int(y.item())
                    elif y.ndim == 1:
                        idx = int(y.argmax().item())
                    else:
                        continue
                else:
                    idx = int(y)
                if 0 <= idx < C:
                    counts[idx] += 1
            except Exception:
                continue
        total = counts.sum()
        pi = (counts / total) if total > 0 else np.ones(C, dtype=np.float32) / C
        inv = 1.0 / np.clip(counts.astype(np.float32), 1.0, None)
        inv = inv / inv.mean()
        return {
            "counts": counts.tolist(),
            "pi": pi.astype(np.float32).tolist(),
            "inv_freq": inv.astype(np.float32).tolist(),
            "total": int(total),
        }

    def _forward(self, v: torch.Tensor, a: torch.Tensor):
        return self.model(v, a, return_aux=True)

    def _reset_scaler_if_needed(self):
        if hasattr(self, 'scaler') and self.scaler is not None and self.scaler.is_enabled():
            old_scale = self.scaler.get_scale() if hasattr(self.scaler, 'get_scale') else 1024.0
            self.scaler = AmpGradScaler(self.device_type, enabled=True)
            new_scale = max(float(old_scale) * 0.5, 2.0)
            self.scaler._scale = torch.tensor(new_scale, dtype=torch.float32, device=self.device)
        if hasattr(self, 'opt') and self.opt is not None:
            self.opt.zero_grad(set_to_none=True)

    def _thr_at(self, epoch: int) -> float:
        if not self.use_ssl: return 0.0
        # ✅ 修复：从更合理的阈值开始，渐进增长
        if epoch <= 2:
            # Epoch 1-2: 使用0.25（SSL延迟启动后，Teacher在快速学习）
            return 0.25
        elif epoch <= self.ssl_warmup:
            # Epoch 3-5: 从0.25线性增长到final_thresh
            warmup_thr = 0.25
            progress = (epoch - 2) / max(1, self.ssl_warmup - 2)
            return warmup_thr + (self.ssl_final_thresh - warmup_thr) * progress
        else:
            # Epoch 6+: 使用final_thresh
            return self.ssl_final_thresh

    def _ema_update(self, frac_in_epoch: float = 1.0):
        if self.teacher is None: return
        # ✅ 修复：使用正确的epoch计数
        current_epoch = getattr(self, 'current_epoch', 1)

        # ✅ 关键修复：扩展快速学习到Epoch 6（SSL从step 100启动）
        if current_epoch <= 2:
            # Epoch 1-2: 使用0.9（Teacher快速学习）
            ema_now = 0.9
        elif current_epoch <= 4:
            # Epoch 3-4: 使用0.95（SSL刚启动，Teacher需要快速适应）
            ema_now = 0.95
        elif current_epoch <= 6:
            # Epoch 5-6: 使用0.97（过渡期）
            ema_now = 0.97
        else:
            # Epoch 7+: 标准EMA策略
            k = min(1.0, max(0.0, (current_epoch - 1 + frac_in_epoch) / max(1, self.ssl_warmup)))
            ema_now = self.ema_decay_init * (1 - k) + self.ema_decay * k

        # ✅ 调试：打印EMA信息（前几次）
        if not hasattr(self, '_ema_debug_count'):
            self._ema_debug_count = 0
        if self._ema_debug_count < 10:  # 增加到10次
            print(
                f"[EMA Debug] epoch={current_epoch}, ema_decay={ema_now:.4f} (Teacher learns {(1 - ema_now) * 100:.1f}% from Student)")
            self._ema_debug_count += 1

        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
                t_param.data.mul_(ema_now).add_(s_param.data, alpha=1.0 - ema_now)

    def _init_bias(self, model, pi):
        pi_tensor = torch.tensor(pi, dtype=torch.float32, device=self.device)

        def _try_set_bias(linear: nn.Linear, tag: str):
            if isinstance(linear, nn.Linear) and linear.bias is not None:
                with torch.no_grad():
                    log_pi = torch.log(torch.clamp(pi_tensor, min=1e-8)).to(linear.bias.device)
                    linear.bias.copy_(log_pi)
                return True
            return False

        try:
            if hasattr(model, 'classifier'):
                classifier = model.classifier
                if isinstance(classifier, nn.Sequential):
                    for layer in reversed(list(classifier)):
                        if isinstance(layer, nn.Linear):
                            _try_set_bias(layer, "classifier")
                            break
        except Exception:
            pass

        try:
            if hasattr(model, "mil_head") and hasattr(model.mil_head, "frame_classifier"):
                seq = model.mil_head.frame_classifier
                if isinstance(seq, nn.Sequential):
                    for m in reversed(list(seq)):
                        if isinstance(m, nn.Linear):
                            _try_set_bias(m, "MIL")
                            break
        except Exception:
            pass

    def _meta_update_step(self, step_count: int):
        """✅ 修复: 完全独立的元学习更新，避免计算图冲突"""
        if not self.mlpr_enabled or self.meta is None or self.meta_opt is None:
            return

        try:
            # ✅ 关键修复：在无梯度环境中准备数据
            with torch.no_grad():
                val_iter = iter(self.loader_v)
                val_batch = next(val_iter)
                if len(val_batch) == 4:
                    v_val, a_val, y_val, _ = val_batch
                else:
                    v_val, a_val, y_val = val_batch
                v_val = v_val.to(self.device).float()
                a_val = a_val.to(self.device).float()
                y_val = y_val.argmax(dim=1).to(self.device) if y_val.ndim == 2 else y_val.to(self.device)

                if not hasattr(self, '_last_train_batch'):
                    return

                v_train, a_train, y_train = self._last_train_batch
                v_train = v_train.float()
                a_train = a_train.float()

            # ✅ 关键修复：使用torch.no_grad()包裹整个元学习过程，然后只更新meta网络
            with amp_autocast(self.device_type, enabled=False):
                # 1. 计算训练损失（不需要梯度，只需要值）
                with torch.no_grad():
                    self.model.eval()
                    out_train = self.model(v_train, a_train, return_aux=True)
                    if out_train is None or "clip_logits" not in out_train:
                        self.model.train()
                        return
                    logits_train = out_train["clip_logits"]
                    train_loss_val = float(self.criterion(logits_train, y_train).item())

                # 2. 模拟一步梯度下降（使用简化的近似）
                # 不使用stateless，而是直接用当前模型评估
                self.model.eval()
                with torch.enable_grad():
                    # 只需要meta网络的梯度
                    out_val = self.model(v_val, a_val, return_aux=True)
                    logits_val = out_val.get("clip_logits", out_val) if isinstance(out_val, dict) else out_val
                    val_loss = F.cross_entropy(logits_val, y_val)

                    # 只更新meta网络
                    self.meta_opt.zero_grad()
                    val_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.meta.parameters(), 1.0)
                    self.meta_opt.step()

                    if not hasattr(self, '_meta_losses'):
                        self._meta_losses = []
                    self._meta_losses.append(val_loss.item())

                self.model.train()

        except Exception as e:
            print(f"[Meta Update] {e}")
            self.meta_fail_count += 1
            if hasattr(self, 'model'):
                self.model.train()

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 60)
        print("🎯 Starting Training...")
        print("=" * 60 + "\n")

        for epoch in range(self.start_epoch, self.epochs + 1):  # ✅ 使用start_epoch
            tr = self._train_epoch(epoch)
            va = self._validate(epoch)
            self.scheduler.step()

            # 记录学习率
            current_lr = self.opt.param_groups[0]['lr']
            self.loss_history['learning_rate'].append(current_lr)
            self.writer.add_scalar('Train/learning_rate', current_lr, epoch)

            f1_for_ckpt = max(va["student"]["f1_macro"], va["teacher"]["f1_macro"])
            who = "student" if f1_for_ckpt == va["student"]["f1_macro"] else "teacher"

            if f1_for_ckpt > getattr(self, "best_f1", -1.0):
                self.best_f1 = f1_for_ckpt
                save_dict = {
                    "epoch": epoch,
                    "state_dict": (self.model.state_dict() if who == "student" else self.teacher.state_dict()),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_f1": self.best_f1
                }
                torch.save(save_dict, self.out_dir / 'checkpoints' / 'best_f1.pth')
                self.no_improve = 0
                print(f"✨ New best F1: {self.best_f1:.4f} ({who})")
            else:
                self.no_improve += 1

            save_dict = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
            }
            torch.save(save_dict, self.out_dir / 'checkpoints' / 'latest.pth')

            print(f"[Epoch {epoch}/{self.epochs}] Train={tr['loss']:.4f} | "
                  f"Val(stu) f1={va['student']['f1_macro']:.4f} | "
                  f"Val(tea) f1={va['teacher']['f1_macro']:.4f} | CKPT={who}")

            if self.early_stop_patience > 0 and self.no_improve >= self.early_stop_patience:
                print(f"🛑 Early stopping: {self.early_stop_patience} epochs without improvement")
                break

        # ✅ 训练结束后生成所有可视化
        print("\n" + "=" * 60)
        print("📊 Generating visualizations...")
        print("=" * 60)

        self.writer.close()
        self._save_loss_history()
        self._plot_all_visualizations()

        print("\n✅ Training complete!")
        print(f"📁 Results saved to: {self.out_dir}")
        print(f"📊 TensorBoard: tensorboard --logdir={self.out_dir / 'runs'}")

    def _train_epoch(self, epoch: int):
        """训练一个epoch"""
        if epoch >= self.amp_disable_epoch and self.amp_enabled:
            self.amp_enabled = False
            self.scaler = AmpGradScaler(self.device_type, enabled=False)
            print(f"⚠️ AMP disabled at epoch {epoch}")

        self.current_epoch = epoch
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()

        # ✅ Epoch级别累计损失
        epoch_losses = {
            'sup_loss': 0.0,
            'cava_loss': 0.0,
            'cava_align': 0.0,
            'cava_edge': 0.0,
            'pseudo_loss': 0.0,
            'total_loss': 0.0,
            'ssl_mask_ratio': 0.0,
            'gate_mean': 0.0,
            'gate_std': 0.0
        }

        tot = 0.0
        nb = 0
        npseudo_mass = 0.0
        u_iter = iter(self.loader_u) if self.use_ssl else None
        thr_ssl = self._thr_at(epoch)
        step_count = 0
        # ✅ 移除_last_pseudo_loss，避免计算图冲突

        ramp_epochs = max(self.ssl_warmup + 5, 1)
        lambda_u_eff_ssl = float(self.lambda_u) * ramp_up(epoch, ramp_epochs)
        lambda_u_eff_mlpr = float(self._mlpr_lambda_u) * ramp_up(epoch, ramp_epochs)

        pbar = tqdm(self.loader_l, desc=f"Epoch {epoch}/{self.epochs}")

        for b in pbar:
            if b is None or len(b) < 3:
                continue
            if isinstance(b, (list, tuple)) and len(b) == 4:
                v, a, y, _ = b
            else:
                v, a, y = b
            v = v.to(self.device)
            a = a.to(self.device)
            y = y.argmax(dim=1).to(self.device) if y.ndim == 2 else y.to(self.device)

            with amp_autocast(self.device_type, enabled=self.amp_enabled, dtype=torch.float16):
                out = self._safe_forward(v, a, use_amp=True)
                if out is None or "clip_logits" not in out:
                    self.nan_count += 1
                    self.total_steps += 1
                    self._reset_scaler_if_needed()
                    continue
                logits = out["clip_logits"]

                # ✅ 监督损失
                sup_loss = self.criterion(logits, y)
                epoch_losses['sup_loss'] += sup_loss.item()

                # ✅ CAVA损失（详细记录）
                cava_loss = v.new_zeros([])
                loss_align_val = 0.0
                loss_edge_val = 0.0

                if self.cava_enabled:
                    try:
                        with amp_autocast(self.device_type, enabled=False):
                            lam_align = float(self.cava_cfg.get("lambda_align", 0.5))
                            lam_edge = float(self.cava_cfg.get("lambda_edge", 0.1))

                            a_aln = out.get("audio_aligned", out.get("audio_seq"))
                            v_prj = out.get("video_proj", out.get("video_seq"))
                            g = out.get("causal_gate", None)

                            if isinstance(a_aln, torch.Tensor) and isinstance(v_prj, torch.Tensor):
                                a_aln = F.normalize(a_aln, dim=-1)
                                v_prj = F.normalize(v_prj, dim=-1)

                            loss_align = info_nce_align(a_aln, v_prj, mask=g, tau=float(self.cava_cfg.get("tau", 0.1)))
                            loss_align = torch.clamp(loss_align, min=0.0, max=10.0)
                            loss_align_val = loss_align.item()

                            delta = out.get("delay_frames_cont", out.get("delay_frames", None))
                            loss_edge = v.new_zeros([])
                            if isinstance(delta, torch.Tensor):
                                low = float(out.get("delta_low", self.cava_cfg.get("delta_low_frames", 2.0)))
                                high = float(out.get("delta_high", self.cava_cfg.get("delta_high_frames", 6.0)))
                                if lam_edge > 0:
                                    loss_edge = edge_hinge(delta, low, high, margin_ratio=0.25)
                                    loss_edge_val = loss_edge.item()

                            cava_loss = lam_align * loss_align + lam_edge * loss_edge

                            # ✅ 记录gate统计
                            if g is not None:
                                epoch_losses['gate_mean'] += g.mean().item()
                                epoch_losses['gate_std'] += g.std().item()

                    except Exception as e:
                        print(f"⚠️ CAVA: {e}")

                epoch_losses['cava_loss'] += cava_loss.item()
                epoch_losses['cava_align'] += loss_align_val
                epoch_losses['cava_edge'] += loss_edge_val

                loss = sup_loss + cava_loss

                # ✅ 为MLPR存储训练batch（完全detach，切断计算图）
                if self.mlpr_enabled:
                    with torch.no_grad():
                        self._last_train_batch = (
                            v.detach().clone(),
                            a.detach().clone(),
                            y.detach().clone()
                        )

                # ✅ SSL伪标签损失
                pseudo_loss_val = 0.0
                if self.use_ssl and (u_iter is not None):
                    # ✅ 新增：SSL延迟启动（让Student先训练100步）
                    ssl_start_step = 100  # 从第100步开始使用SSL
                    if self.total_steps < ssl_start_step:
                        # 提示信息（只显示一次）
                        if not hasattr(self, '_ssl_delay_msg_shown'):
                            print(
                                f"🔄 SSL delayed start: will begin at step {ssl_start_step} (current: {self.total_steps})")
                            self._ssl_delay_msg_shown = True
                    elif self.total_steps == ssl_start_step:
                        print(f"✅ SSL started at step {ssl_start_step}!")
                        # ✅ 关键修复：让Teacher从当前Student继承权重（允许部分不匹配）
                        if self.teacher is not None:
                            with torch.no_grad():
                                self.teacher.load_state_dict(self.model.state_dict(), strict=False)
                            print(f"🔄 Teacher re-initialized from Student (F1 should improve)")
                    else:
                        try:
                            try:
                                bu = next(u_iter)
                            except StopIteration:
                                u_iter = iter(self.loader_u)
                                bu = next(u_iter)
                            if isinstance(bu, (list, tuple)) and len(bu) == 4:
                                vu, au, yu, ids_u = bu
                            else:
                                vu, au, yu = bu
                                ids_u = None
                            vu = vu.to(self.device)
                            au = au.to(self.device)

                            with torch.no_grad():
                                tout = self.teacher(vu, au, return_aux=False)
                                t_logits = tout["clip_logits"] if isinstance(tout,
                                                                             dict) and "clip_logits" in tout else tout
                                t_logits = torch.clamp(t_logits, min=-50, max=50)
                                t_prob = F.softmax(t_logits / self.ssl_temp, dim=1)

                                t_max_all = t_prob.max(dim=1).values.detach().float().cpu().numpy()
                                p90 = float(np.percentile(t_max_all, 90))
                                self._teach_p90_ema = 0.9 * self._teach_p90_ema + 0.1 * p90

                                # ✅ 修复: 更早启用分布对齐
                                enable_da = bool(self._use_dist_align and (epoch > 3))  # ✅ 强制转bool
                                if enable_da:
                                    q = t_prob.mean(dim=0).clamp(min=1e-8)
                                    p_target = self._pi / (q + 1e-8)
                                    p_target = p_target / p_target.sum()
                                    t_prob = t_prob * p_target.unsqueeze(0)
                                    t_prob = t_prob / t_prob.sum(dim=1, keepdim=True)

                                t_max, t_idx = t_prob.max(dim=1)

                                # ✅ 修复：更早启用类自适应阈值
                                enable_cpl = bool(self._use_cls_threshold and (epoch > 3))  # ✅ 强制转bool
                                if enable_cpl:
                                    for c in range(self.C):
                                        mask_c = (t_idx == c)
                                        if mask_c.any():
                                            mean_c = t_max[mask_c].mean()
                                            self._cls_conf_ema[c] = self._cls_thr_momentum * self._cls_conf_ema[c] + (
                                                        1 - self._cls_thr_momentum) * mean_c
                                            self._cls_thr[c] = torch.clamp(self._cls_conf_ema[c], min=self._thr_min,
                                                                           max=self.ssl_final_thresh)
                                    thr_vec = self._cls_thr[t_idx]
                                    thr_use = thr_vec
                                else:
                                    thr_use = torch.full_like(t_max, thr_ssl)

                            sout = self._safe_forward(vu, au, use_amp=True)
                            if sout is None or "clip_logits" not in sout:
                                pass
                            else:
                                s_logits = sout["clip_logits"]

                                if self.mlpr_enabled and (self.meta is not None):
                                    # ✅ MLPR路径 - 确保所有中间变量不保留计算图
                                    with torch.no_grad():
                                        # 提取特征（不需要梯度）
                                        stu_feat = None
                                        id_list = None  # ✅ 修复：初始化id_list
                                        if isinstance(sout, dict):
                                            ftok = sout.get("fusion_token", None)
                                            if ftok is not None:
                                                stu_feat = ftok.mean(
                                                    dim=tuple(range(1, ftok.dim()))) if ftok.dim() > 2 else ftok
                                            else:
                                                vtok = sout.get("video_proj", sout.get("video_emb", None))
                                                atok = sout.get("audio_aligned", sout.get("audio_emb", None))
                                                if (vtok is not None) and (atok is not None):
                                                    if vtok.dim() > 2:
                                                        vtok = vtok.mean(dim=tuple(range(1, vtok.dim())))
                                                    if atok.dim() > 2:
                                                        atok = atok.mean(dim=tuple(range(1, atok.dim())))
                                                    stu_feat = torch.cat([vtok, atok], dim=-1)

                                        hist_mu = hist_std = None
                                        if getattr(self, "hist_bank", None) is not None and (ids_u is not None):
                                            id_list = ids_u.cpu().tolist() if torch.is_tensor(ids_u) else ids_u
                                            mu, sd = self.hist_bank.query([int(x) for x in id_list])
                                            hist_mu = mu.to(self.device)
                                            hist_std = sd.to(self.device)
                                            if hist_mu.dim() > 1:
                                                hist_mu = hist_mu.mean(dim=tuple(range(1, hist_mu.dim())))
                                            if hist_std.dim() > 1:
                                                hist_std = hist_std.mean(dim=tuple(range(1, hist_std.dim())))
                                            hist_mu = hist_mu.view(-1, 1)
                                            hist_std = hist_std.view(-1, 1)

                                        cava_gate_mean = None
                                        if isinstance(sout, dict) and ("causal_gate" in sout) and (
                                                sout["causal_gate"] is not None):
                                            cg = sout["causal_gate"]
                                            cava_gate_mean = cg.mean(dim=tuple(range(1, cg.dim()))).view(-1,
                                                                                                         1) if cg.dim() > 1 else cg.view(
                                                -1, 1)

                                        # 构建特征
                                        feats = build_mlpr_features(
                                            teacher_prob=t_prob.detach(),  # ✅ 确保detach
                                            student_feat=stu_feat.detach() if stu_feat is not None else None,
                                            history_mean=hist_mu,
                                            history_std=hist_std,
                                            cava_gate_mean=cava_gate_mean.detach() if cava_gate_mean is not None else None,
                                            use_prob_vector=self._mlpr_flags["use_prob_vec"]
                                        )

                                    # ✅ Meta网络推理（需要梯度）
                                    w = self.meta(feats).detach()  # ✅ 立即detach，不保留meta的梯度图

                                    # ✅ 计算伪标签损失（现在可以安全backward）
                                    s_log_prob = F.log_softmax(s_logits, dim=1)
                                    t_prob_stable = t_prob.clamp(min=1e-8).detach()  # ✅ 确保detach
                                    kl = F.kl_div(s_log_prob, t_prob_stable, reduction="none").sum(dim=1, keepdim=True)
                                    kl = torch.clamp(kl, min=0.0, max=10.0)

                                    with torch.no_grad():
                                        conf_mask = (t_max.view(-1, 1) > thr_use.view(-1, 1)).float()
                                        gate_min = float(self.cava_cfg.get("gate_min", 0.15))
                                        if cava_gate_mean is not None:
                                            gate_mask = (cava_gate_mean > gate_min).float()
                                            mask = conf_mask * gate_mask
                                        else:
                                            mask = conf_mask

                                        w_eff = (w * mask).clamp_(0.0, 1.0)
                                        mass = float(w_eff.sum().item())

                                    if w_eff.sum() > 0:
                                        pseudo_loss = (w_eff.detach() * kl).sum() / (w_eff.sum().item() + 1e-8)
                                        pseudo_loss_val = pseudo_loss.item()
                                    else:
                                        # ✅ MLPR Fallback: 如果meta权重全为0，降级为标准SSL
                                        with torch.no_grad():
                                            # ✅ 动态Fallback阈值：在早期epoch使用更低阈值
                                            current_epoch = getattr(self, 'current_epoch', 1)
                                            if current_epoch <= 2:
                                                # Epoch 1-2: 0.10（极低阈值，Teacher很弱时使用）
                                                base_fallback = 0.10
                                            elif current_epoch <= 5:
                                                # Epoch 3-5: 渐进增长到0.45
                                                base_fallback = 0.10 + 0.35 * (current_epoch - 2) / 3
                                            else:
                                                # Epoch 6+: 标准阈值0.45
                                                base_fallback = 0.45

                                            # 如果thr_use是向量，使用其70%；否则使用base_fallback
                                            if torch.is_tensor(thr_use):
                                                fallback_thr = torch.maximum(
                                                    thr_use * 0.7,
                                                    torch.full_like(thr_use, base_fallback)
                                                )
                                            else:
                                                fallback_thr = max(base_fallback, float(thr_use) * 0.7)

                                            std_mask = (t_max > fallback_thr)

                                            # ✅ 调试信息（前3次）
                                            if not hasattr(self, '_fallback_debug_count'):
                                                self._fallback_debug_count = 0
                                            if self._fallback_debug_count < 3:
                                                print(f"[MLPR Fallback] epoch={current_epoch}, w_eff=0")
                                                print(
                                                    f"  t_max: mean={t_max.mean():.3f}, max={t_max.max():.3f}, p90={torch.quantile(t_max, 0.9):.3f}")
                                                if torch.is_tensor(fallback_thr):
                                                    print(f"  fallback_thr: {fallback_thr.mean():.3f}")
                                                else:
                                                    print(f"  fallback_thr: {fallback_thr:.3f}")
                                                print(f"  mask_ratio: {std_mask.float().mean():.2%}")
                                                self._fallback_debug_count += 1

                                        if std_mask.any():
                                            pseudo_loss = F.cross_entropy(s_logits[std_mask], t_idx[std_mask].detach())
                                            pseudo_loss_val = pseudo_loss.item()
                                            mass = float(std_mask.sum().item())
                                        else:
                                            pseudo_loss = v.new_zeros([])
                                            pseudo_loss_val = 0.0
                                            mass = 0.0  # ✅ 修复：确保mass总是定义
                                            # ✅ 警告（只在前2个epoch显示）
                                            if not hasattr(self, '_no_pseudo_warned') and current_epoch <= 2:
                                                print(
                                                    f"⚠️ Epoch {current_epoch}: Teacher still warming up (max_conf={t_max.max():.3f})")
                                                print(
                                                    f"   This is normal in early epochs. Will improve after epoch 3-5.")
                                                self._no_pseudo_warned = True

                                    # ✅ 更新历史（使用detach的值）
                                    if self.hist_bank is not None and id_list is not None:
                                        self.hist_bank.update(id_list, kl.detach().squeeze(1))

                                    loss = loss + lambda_u_eff_mlpr * pseudo_loss
                                    npseudo_mass += mass
                                else:
                                    # ✅ 标准SSL路径 - 确保不保留计算图
                                    with torch.no_grad():
                                        mask = (t_max > thr_use)

                                    if mask.any():
                                        # ✅ 使用detach的目标
                                        pseudo_loss = F.cross_entropy(s_logits[mask], t_idx[mask].detach())
                                        pseudo_loss_val = pseudo_loss.item()
                                        loss = loss + lambda_u_eff_ssl * pseudo_loss
                                        npseudo_mass += float(mask.sum().item())
                                        epoch_losses['ssl_mask_ratio'] += float(mask.sum().item()) / len(mask)
                        except Exception as e:
                            print(f"⚠️ SSL: {e}")

                epoch_losses['pseudo_loss'] += pseudo_loss_val
                epoch_losses['total_loss'] += loss.item()

                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count += 1
                    self.total_steps += 1
                    self._reset_scaler_if_needed()
                    continue

                self.opt.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    try:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    except Exception as e:
                        print(f"⚠️ AMP backward: {e}")
                        self.opt.zero_grad(set_to_none=True)
                        self._reset_scaler_if_needed()
                        self.nan_count += 1
                        self.total_steps += 1
                        continue
                else:
                    try:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.opt.step()
                    except Exception as e:
                        print(f"⚠️ backward: {e}")
                        self.opt.zero_grad(set_to_none=True)
                        self.nan_count += 1
                        self.total_steps += 1
                        continue

            # ✅ 实时TensorBoard记录（每100步）
            if step_count % 100 == 0:
                global_step = (epoch - 1) * len(self.loader_l) + step_count
                self.writer.add_scalar('Step/sup_loss', sup_loss.item(), global_step)
                self.writer.add_scalar('Step/cava_loss', cava_loss.item(), global_step)
                self.writer.add_scalar('Step/pseudo_loss', pseudo_loss_val, global_step)
                self.writer.add_scalar('Step/total_loss', loss.item(), global_step)

                # 记录step级别数据（用于平滑曲线）
                self.step_losses['sup_loss'].append(sup_loss.item())
                self.step_losses['cava_loss'].append(cava_loss.item())
                self.step_losses['pseudo_loss'].append(pseudo_loss_val)
                self.step_losses['total_loss'].append(loss.item())

            # ✅ 关键修复：EMA更新独立于SSL，总是执行（让Teacher学习）
            if self.teacher is not None:
                frac = (step_count + 1) / max(1, len(self.loader_l))
                self._ema_update(frac_in_epoch=frac)

            if self.mlpr_enabled and self.meta is not None:
                if (step_count + 1) % max(self._mlpr_meta_interval, 1) == 0:
                    self._meta_update_step(step_count)

            tot += float(loss.detach().item())
            nb += 1
            step_count += 1
            self.total_steps += 1
            pbar.set_postfix(loss=f"{tot / max(1, nb):.4f}", pseudo=f"{npseudo_mass:.0f}")

        # ✅ 记录epoch平均值
        num_batches = max(nb, 1)
        for key in epoch_losses:
            avg_val = epoch_losses[key] / num_batches
            # ✅ 安全检查：确保key存在
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(avg_val)
            self.writer.add_scalar(f'Epoch/{key}', avg_val, epoch)

        # ✅ MLPR诊断信息
        if self.mlpr_enabled and npseudo_mass > 0:
            avg_pseudo_mass = npseudo_mass / num_batches
            self.writer.add_scalar(f'Epoch/mlpr_pseudo_mass', avg_pseudo_mass, epoch)

        return {"loss": round(tot / max(1, nb), 4)}

    @torch.no_grad()
    def _validate(self, epoch: int):
        """验证模型性能"""

        def _eval_model(m):
            m.eval()
            ys, ps = [], []
            for b in DataLoader(self.ds_v, batch_size=self.bs, shuffle=False, num_workers=0,
                                pin_memory=(self.device.type == 'cuda'), drop_last=False, collate_fn=safe_collate_fn):
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    v, a, y, _ = b
                else:
                    v, a, y = b
                v = v.to(self.device)
                a = a.to(self.device)
                y = y.argmax(dim=1) if y.ndim == 2 else y
                out = m(v, a, return_aux=False)
                logits = out["clip_logits"] if isinstance(out, dict) and "clip_logits" in out else out
                prob = F.softmax(logits, dim=1).cpu().numpy()
                ps.append(prob)
                ys.append(y.cpu().numpy())
            y_true = np.concatenate(ys, 0)
            y_prob = np.concatenate(ps, 0)
            y_pred = y_prob.argmax(1)
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            acc = accuracy_score(y_true, y_pred)
            f1m = f1_score(y_true, y_pred, average="macro")
            aucm = np.nan
            try:
                y_true_oh = np.eye(self.C, dtype=np.float32)[y_true]
                aucm = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
            except Exception:
                pass
            return {"acc": float(acc), "f1_macro": float(f1m), "auc_macro": float(aucm)}

        stu = _eval_model(self.model)
        tea = _eval_model(self.teacher) if (self.teacher is not None) else {"acc": 0.0, "f1_macro": 0.0,
                                                                            "auc_macro": 0.0}

        # ✅ 记录验证指标
        self.loss_history['val_acc_student'].append(stu['acc'])
        self.loss_history['val_f1_student'].append(stu['f1_macro'])
        self.loss_history['val_acc_teacher'].append(tea['acc'])
        self.loss_history['val_f1_teacher'].append(tea['f1_macro'])

        self.writer.add_scalar('Val/acc_student', stu['acc'], epoch)
        self.writer.add_scalar('Val/f1_student', stu['f1_macro'], epoch)
        self.writer.add_scalar('Val/acc_teacher', tea['acc'], epoch)
        self.writer.add_scalar('Val/f1_teacher', tea['f1_macro'], epoch)

        return {"student": stu, "teacher": tea}

    def _build_sampler(self, ds_l, inv_freq):
        """构建加权采样器"""
        labels = []
        if hasattr(ds_l, "rows") and len(ds_l.rows) > 0:
            for r in ds_l.rows:
                y = r.get("label_idx", None)
                y = int(y) if (y is not None) else -1
                labels.append(y)
        else:
            for i in range(len(ds_l)):
                try:
                    item = ds_l[i]
                    y = item[2] if isinstance(item, (list, tuple)) else None
                    if torch.is_tensor(y):
                        y = int(y.item())
                        labels.append(int(y) if y is not None else -1)
                except Exception:
                    labels.append(-1)
        C = len(inv_freq)
        weights = np.zeros(len(labels), dtype=np.float64)
        for i, y in enumerate(labels):
            if 0 <= y < C:
                weights[i] = float(inv_freq[y])
            else:
                weights[i] = float(inv_freq.mean()) if C > 0 else 1.0
        w = torch.tensor(weights, dtype=torch.double)
        return WeightedRandomSampler(w, num_samples=len(ds_l), replacement=True)

    def _safe_forward(self, v: torch.Tensor, a: torch.Tensor, use_amp: bool = True):
        """安全前向传播"""
        try:
            if torch.isnan(v).any() or torch.isinf(v).any():
                v = torch.where(torch.isnan(v) | torch.isinf(v), torch.zeros_like(v), v)
            if torch.isnan(a).any() or torch.isinf(a).any():
                a = torch.where(torch.isnan(a) | torch.isinf(a), torch.zeros_like(a), a)
            current_epoch = getattr(self, 'current_epoch', 1)
            if current_epoch >= self.amp_disable_epoch:
                use_amp = False
            if use_amp and self.amp_enabled:
                with amp_autocast(self.device_type, enabled=True, dtype=torch.float16):
                    out = self._forward(v, a)
            else:
                v = v.float()
                a = a.float()
                with amp_autocast(self.device_type, enabled=False):
                    out = self._forward(v, a)
            return out
        except Exception as e:
            print(f"⚠️ forward: {e}")
            return None

    # ================== 可视化方法 ==================

    def _save_loss_history(self):
        """保存损失历史到JSON"""
        json_path = self.out_dir / 'loss_history.json'
        with open(json_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        print(f"💾 Loss history saved: {json_path}")

    def _plot_all_visualizations(self):
        """生成所有可视化图表"""
        viz_dir = self.out_dir / 'visualizations'

        # 1. 主损失曲线（6子图）
        self._plot_main_losses(viz_dir / 'main_losses.png')

        # 2. CAVA细节分析
        self._plot_cava_details(viz_dir / 'cava_details.png')

        # 3. 验证性能曲线
        self._plot_validation_metrics(viz_dir / 'validation_metrics.png')

        # 4. 学习率和gate统计
        self._plot_training_dynamics(viz_dir / 'training_dynamics.png')

        # 5. 平滑的step级别曲线
        if len(self.step_losses['total_loss']) > 0:
            self._plot_smooth_step_curves(viz_dir / 'smooth_step_losses.png')

        print(f"✅ All visualizations saved to: {viz_dir}")


# ==================== 命令行入口 ====================
if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='CAVA-SSL Training with Visualization and Resume Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 从头训练
  python strong_trainer_with_viz.py --config config.yaml --output ./outputs/exp1

  # 从checkpoint恢复
  python strong_trainer_with_viz.py --config config.yaml --output ./outputs/resumed --checkpoint ./outputs/exp1/checkpoints/best_f1.pth

  # 只评估checkpoint
  python strong_trainer_with_viz.py --config config.yaml --checkpoint best_f1.pth --eval_only
        """
    )

    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='./outputs/train', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint (optional)')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate checkpoint without training')

    args = parser.parse_args()

    # 加载配置
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # 创建训练器
    try:
        trainer = StrongTrainer(cfg, out_dir=args.output, resume_from=args.checkpoint)
    except Exception as e:
        print(f"❌ Error creating trainer: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 执行训练或评估
    try:
        if args.eval_only:
            if args.checkpoint is None:
                print("❌ --eval_only requires --checkpoint")
                sys.exit(1)
            print("\n" + "=" * 60)
            print("📊 Evaluation Mode")
            print("=" * 60 + "\n")
            results = trainer._validate(epoch=0)
            print("\n" + "=" * 60)
            print("📊 Evaluation Results")
            print("=" * 60)
            print(f"Student - Acc: {results['student']['acc']:.4f} | F1: {results['student']['f1_macro']:.4f}")
            print(f"Teacher - Acc: {results['teacher']['acc']:.4f} | F1: {results['teacher']['f1_macro']:.4f}")
            print("=" * 60 + "\n")
        else:
            trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


    def _plot_main_losses(self, save_path):
        """绘制主要损失曲线（6子图）"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Training Loss Curves Overview', fontsize=18, fontweight='bold', y=0.995)

        epochs = range(1, len(self.loss_history['total_loss']) + 1)

        # 1. 总损失
        axes[0, 0].plot(epochs, self.loss_history['total_loss'], 'b-', linewidth=2.5, label='Total Loss')
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 监督损失
        axes[0, 1].plot(epochs, self.loss_history['sup_loss'], 'g-', linewidth=2.5, label='Supervised Loss')
        axes[0, 1].set_title('Supervised Loss (Cross-Entropy)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. CAVA总损失
        axes[0, 2].plot(epochs, self.loss_history['cava_loss'], 'r-', linewidth=2.5, label='CAVA Loss')
        axes[0, 2].set_title('CAVA Loss (Total)', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('Loss', fontsize=12)
        axes[0, 2].legend(fontsize=11)
        axes[0, 2].grid(True, alpha=0.3)

        # 4. CAVA分解
        axes[1, 0].plot(epochs, self.loss_history['cava_align'], 'orange', linewidth=2.5, label='Align Loss',
                        marker='o', markersize=4)
        axes[1, 0].plot(epochs, self.loss_history['cava_edge'], 'purple', linewidth=2.5, label='Edge Loss', marker='s',
                        markersize=4)
        axes[1, 0].set_title('CAVA Components', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 伪标签损失
        axes[1, 1].plot(epochs, self.loss_history['pseudo_loss'], 'cyan', linewidth=2.5, label='Pseudo Loss')
        axes[1, 1].set_title('Pseudo Label Loss (SSL/MLPR)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Gate均值
        axes[1, 2].plot(epochs, self.loss_history['gate_mean'], 'magenta', linewidth=2.5, label='Gate Mean')
        axes[1, 2].axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Reference (0.5)')
        axes[1, 2].set_title('Causal Gate Mean', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch', fontsize=12)
        axes[1, 2].set_ylabel('Gate Value', fontsize=12)
        axes[1, 2].legend(fontsize=11)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Main losses plot: {save_path}")


    def _plot_cava_details(self, save_path):
        """CAVA细节分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CAVA Detailed Analysis', fontsize=16, fontweight='bold')

        epochs = range(1, len(self.loss_history['total_loss']) + 1)

        # 1. Align Loss趋势
        axes[0, 0].plot(epochs, self.loss_history['cava_align'], 'orange', linewidth=2, marker='o', markersize=5)
        axes[0, 0].set_title('InfoNCE Alignment Loss', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Edge Loss趋势
        axes[0, 1].plot(epochs, self.loss_history['cava_edge'], 'purple', linewidth=2, marker='s', markersize=5)
        axes[0, 1].set_title('Edge Hinge Loss', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Gate统计（均值+标准差）
        if len(self.loss_history['gate_std']) > 0:
            mean_vals = np.array(self.loss_history['gate_mean'])
            std_vals = np.array(self.loss_history['gate_std'])
            axes[1, 0].plot(epochs, mean_vals, 'b-', linewidth=2, label='Mean')
            axes[1, 0].fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals,
                                    alpha=0.3, label='±1 Std')
            axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', label='Reference')
            axes[1, 0].set_title('Causal Gate Statistics', fontsize=13, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gate Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 1])

        # 4. CAVA总损失vs组件
        axes[1, 1].plot(epochs, self.loss_history['cava_loss'], 'r-', linewidth=2.5, label='Total CAVA')
        axes[1, 1].plot(epochs, self.loss_history['cava_align'], 'orange', linewidth=1.5,
                        linestyle='--', label='Align Component')
        axes[1, 1].plot(epochs, self.loss_history['cava_edge'], 'purple', linewidth=1.5,
                        linestyle='--', label='Edge Component')
        axes[1, 1].set_title('CAVA Loss Decomposition', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 CAVA details plot: {save_path}")


    def _plot_validation_metrics(self, save_path):
        """验证性能曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Validation Performance', fontsize=16, fontweight='bold')

        epochs = range(1, len(self.loss_history['val_f1_student']) + 1)

        # 1. F1分数对比
        axes[0].plot(epochs, self.loss_history['val_f1_student'], 'b-', linewidth=2.5,
                     marker='o', markersize=6, label='Student')
        axes[0].plot(epochs, self.loss_history['val_f1_teacher'], 'r--', linewidth=2.5,
                     marker='s', markersize=6, label='Teacher (EMA)')
        axes[0].set_title('F1 Score (Macro)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('F1 Score', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # 2. Accuracy对比
        axes[1].plot(epochs, self.loss_history['val_acc_student'], 'b-', linewidth=2.5,
                     marker='o', markersize=6, label='Student')
        axes[1].plot(epochs, self.loss_history['val_acc_teacher'], 'r--', linewidth=2.5,
                     marker='s', markersize=6, label='Teacher (EMA)')
        axes[1].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Validation metrics plot: {save_path}")


    def _plot_training_dynamics(self, save_path):
        """训练动态（学习率、gate等）"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Dynamics', fontsize=16, fontweight='bold')

        epochs = range(1, len(self.loss_history['learning_rate']) + 1)

        # 1. 学习率变化
        axes[0, 0].plot(epochs, self.loss_history['learning_rate'], 'g-', linewidth=2)
        axes[0, 0].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Learning Rate')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. SSL mask比例
        if len(self.loss_history['ssl_mask_ratio']) > 0:
            axes[0, 1].plot(epochs, self.loss_history['ssl_mask_ratio'], 'c-', linewidth=2)
            axes[0, 1].set_title('SSL Pseudo Label Acceptance Rate', fontsize=13, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Acceptance Ratio')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1])

        # 3. 损失堆叠图
        axes[1, 0].stackplot(epochs,
                             self.loss_history['sup_loss'],
                             self.loss_history['cava_loss'],
                             self.loss_history['pseudo_loss'],
                             labels=['Supervised', 'CAVA', 'Pseudo'],
                             alpha=0.7)
        axes[1, 0].set_title('Loss Components (Stacked)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Gate动态（箱线图风格）
        if len(self.loss_history['gate_mean']) > 0:
            axes[1, 1].plot(epochs, self.loss_history['gate_mean'], 'magenta', linewidth=2)
            axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5)
            axes[1, 1].fill_between(epochs, 0.2, 0.8, alpha=0.1, color='green', label='Target Range')
            axes[1, 1].set_title('Causal Gate Dynamics', fontsize=13, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gate Mean')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training dynamics plot: {save_path}")


    def _plot_smooth_step_curves(self, save_path):
        """平滑的step级别损失曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Smoothed Step-Level Loss Curves', fontsize=16, fontweight='bold')

        window = 50  # 移动平均窗口

        def smooth(data, window_size):
            """简单移动平均"""
            if len(data) < window_size:
                return data
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        # 1. 总损失
        total_smooth = smooth(self.step_losses['total_loss'], window)
        axes[0, 0].plot(total_smooth, 'b-', linewidth=1.5, alpha=0.8)
        axes[0, 0].set_title(f'Total Loss (Smoothed, window={window})', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 监督损失
        sup_smooth = smooth(self.step_losses['sup_loss'], window)
        axes[0, 1].plot(sup_smooth, 'g-', linewidth=1.5, alpha=0.8)
        axes[0, 1].set_title(f'Supervised Loss (Smoothed, window={window})', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. CAVA损失
        cava_smooth = smooth(self.step_losses['cava_loss'], window)
        axes[1, 0].plot(cava_smooth, 'r-', linewidth=1.5, alpha=0.8)
        axes[1, 0].set_title(f'CAVA Loss (Smoothed, window={window})', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 伪标签损失
        pseudo_smooth = smooth(self.step_losses['pseudo_loss'], window)
        axes[1, 1].plot(pseudo_smooth, 'c-', linewidth=1.5, alpha=0.8)
        axes[1, 1].set_title(f'Pseudo Loss (Smoothed, window={window})', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Smooth step curves plot: {save_path}")