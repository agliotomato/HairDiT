"""
Trainer for SD3.5 ControlNet hair region generation.

Holds:
  - HairControlNet (trainable): ControlNet + MatteCNN + learned null embeddings
  - SD3Transformer2DModel (frozen): generates velocity predictions via ControlNet residuals
  - VAEWrapper SD3.5 (frozen): 16-channel latents
  - FlowMatchEulerDiscreteScheduler: flow matching noise schedule

Phase 1 (pretrain):
  - Dataset: unbraid (3K samples), 200 epochs
  - Only HairControlNet.parameters() trained
  - Transformer fully frozen

Phase 2 (finetune):
  - Dataset: braid (1K samples), 100 epochs
  - Load Phase 1 checkpoint
  - HairControlNet trained with lower lr
  - Transformer still frozen

Training step:
  1. target → VAE encode → latents (16ch, 64x64)
  2. Sample sigma from logit-normal distribution
  3. noisy_latents = (1-sigma)*latents + sigma*noise
  4. HairControlNet(noisy_latents, sketch, matte, timesteps) → block_samples, null_hs, null_pooled
  5. transformer(noisy_latents, null_hs, null_pooled, timesteps, block_samples) → v_pred
     (timesteps = sigma * 1000 — SD3.5 사전학습 규약. sigma 원값은 노이징/x0 복원에만 사용)
  6. v_target = noise - latents
  7. loss = flow_loss + lpips + edge
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import build_augmentation_pipeline
from src.data.dataset import HairRegionDataset, StratifiedBatchSampler, select_fixed_indices
from src.data.utils import resize_matte_to_latent
from src.models.controlnet_sd35 import HairControlNet, gate_block_samples, transformer_forward_full_residual
from src.models.vae_wrapper import VAEWrapper
from src.training.ema import EMAModel
from src.training.losses import HairLoss

# held-out 16+16 평가셋 선택 + perceptual val 생성 noise 공통 seed.
# 8:8 stratified sampler의 sample_seed:0 관례와 통일 ([0724] planning §4-1/§4-2).
PERCEPTUAL_VAL_SEED = 0
PERCEPTUAL_VAL_N = 16
PERCEPTUAL_VAL_STEPS = 20


def _flatten_config(cfg: dict, prefix: str = "") -> dict:
    """Nested dict → flat dict with dot-separated keys (TensorBoard hparam 호환)."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_config(v, key))
        elif isinstance(v, (int, float, str, bool)):
            out[key] = v
        elif v is None:
            out[key] = "null"
        else:
            out[key] = str(v)  # list 등은 문자열로
    return out


def _to_u8_hwc(t: torch.Tensor) -> list[np.ndarray]:
    """(B,C,H,W) float [0,1] → (H,W,C) uint8 numpy 배열 리스트 (scripts/eval_metrics.py 지표 입력용)."""
    arr = (t.float().clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


def _to_u8_hw(t: torch.Tensor) -> list[np.ndarray]:
    """(B,1,H,W) float [0,1] → (H,W) uint8 numpy 배열 리스트."""
    arr = (t.float().clamp(0, 1).detach().cpu().squeeze(1).numpy() * 255).round().astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


class Trainer:
    """
    Unified trainer for Phase 1 (pretrain) and Phase 2 (finetune).

    Args:
        config: dict loaded from YAML config file
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.phase = config["training"]["phase"]
        assert self.phase in ("pretrain", "finetune"), f"Unknown phase: {self.phase}"

        # Inverse task self-distillation
        self.inverse_mode = config["training"].get("mode", "forward") == "inverse"
        self.schedule     = config["training"].get("schedule", "none")
        self.gate_alpha   = config["training"].get("gate_alpha", 1.0)   # PDF alpha gate (Eq. 9)
        self.w_cycle      = config["training"]["loss_weights"].get("cycle", 0.0)
        self.cycle_start  = config["training"].get("cycle_start", 9999)
        self.w_sketch_dec = config["training"]["loss_weights"].get("sketch_decoder", 0.0)
        self.forward_controlnet: Optional[HairControlNet] = None

        # wandb는 이 env(protobuf 7)와 호환되지 않을 수 있으므로 import 가능할 때만 사용
        self._trackers = ["tensorboard"]
        try:
            import wandb  # noqa: F401
            self._trackers.append("wandb")
        except Exception as e:  # pragma: no cover - 환경 의존
            print(f"[trainer] wandb 비활성화 (import 실패: {e}). tensorboard만 사용.")

        self.accelerator = Accelerator(
            mixed_precision=config["training"].get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 2),
            log_with=self._trackers,
            project_dir=config["checkpointing"]["output_dir"],
        )

        self._setup_models()
        self._setup_data()
        self._setup_perceptual_eval()
        self._setup_optimizer()
        self._prepare_accelerator()

        self.loss_fn = HairLoss(
            phase=self.phase,
            w_flow=config["training"]["loss_weights"].get("flow", 1.0),
            w_lpips=config["training"]["loss_weights"].get("lpips", 0.1),
            w_edge=config["training"]["loss_weights"].get("edge", 0.0),
            lpips_warmup_frac=config["training"]["loss_weights"].get("lpips_warmup_frac", 0.3),
            scale_sync=config["training"]["loss_weights"].get("scale_sync", True),
            s_min=config["training"]["loss_weights"].get("s_min", 20.0),
            s_max=config["training"]["loss_weights"].get("s_max", 120.0),
        ).to(self.accelerator.device)

        self.ema = EMAModel(
            self.accelerator.unwrap_model(self.controlnet),
            decay=config["training"].get("ema_decay", 0.9999),
        )

        self.output_dir = Path(config["checkpointing"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracker 초기화 (TensorBoard + wandb)
        wcfg = config.get("wandb", {})
        run_name = f"{config['training']['phase']}_{config['training']['dataset']}"
        tags = wcfg.get("tags", []) + [config["training"]["phase"], config["training"]["dataset"]]
        self.accelerator.init_trackers(
            project_name=wcfg.get("project", "hair-dit"),
            config=_flatten_config(config),  # TensorBoard은 primitive만 허용
            init_kwargs={
                "wandb": {
                    "name": run_name,
                    "entity": wcfg.get("entity") or None,
                    "tags": tags,
                    "dir": str(self.output_dir),
                    "config": config,  # wandb에는 원본 전달
                },
            },
        )

        # Flow matching timestep sampling hyperparams
        self.logit_mean = config["training"].get("logit_mean", 0.0)
        self.logit_std  = config["training"].get("logit_std",  1.0)

        self.global_step   = 0
        self.best_val_loss = float("inf")
        self.start_epoch   = 0
        self._current_epoch = 0

        # perceptual val 조기종료 / 정점 ckpt 선정 상태 ([0724] planning §4-3)
        self._best_unbraid = float("inf")
        self._best_select  = float("inf")
        self._patience     = 0

        self._restore_training_state()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_models(self):
        cfg = self.cfg
        model_id = cfg["model"]["model_id"]
        local_files_only = cfg.get("local_files_only", False)

        # Frozen SD3.5 VAE (bfloat16)
        self.vae = VAEWrapper.from_pretrained(
            model_id=model_id,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        self.vae.eval()

        # Frozen SD3.5 Transformer
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        self.transformer.eval()

        # Trainable HairControlNet (builds ControlNet from transformer weights)
        num_layers = cfg["model"].get("num_controlnet_layers", 12)
        self.controlnet = HairControlNet(
            model_id=model_id,
            vae=self.vae,
            num_layers=num_layers,
            local_files_only=local_files_only,
            use_sketch_decoder=cfg["model"].get("use_sketch_decoder", False),
            zero_matte_cond=cfg["model"].get("zero_matte_cond", False),
            zero_matte_feat=cfg["model"].get("zero_matte_feat", False),
            zero_raw_matte=cfg["model"].get("zero_raw_matte", False),
            num_extra_conditioning_channels=cfg["model"].get("num_extra_conditioning_channels", 16),
            matte_bias_zero_init=cfg["model"].get("matte_bias_zero_init", True),
            use_matte_scale=cfg["model"].get("use_matte_scale", True),
        )

        # Gradient checkpointing: saves ~40% activation memory at ~20% compute cost
        # Required when LPIPS is active: VAE decode computation graph adds significant memory
        if cfg["training"].get("gradient_checkpointing", True):
            self.transformer.enable_gradient_checkpointing()
            self.controlnet.controlnet.enable_gradient_checkpointing()
            self.vae.vae.enable_gradient_checkpointing()

        # Flow matching scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # perceptual val 전용 별도 scheduler 인스턴스. set_timesteps()가 self.sigmas/timesteps를
        # num_inference_steps 길이로 덮어써 학습용 _sample_sigmas()의 인덱싱을 깨뜨리므로,
        # run_sampling_batched(§4-2)에는 반드시 이 인스턴스를 넘긴다.
        self.sample_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # Inverse mode: load frozen forward ControlNet for cycle self-distillation
        if self.inverse_mode and self.w_cycle > 0.0:
            fwd_ckpt = cfg["training"].get("forward_checkpoint")
            if fwd_ckpt and Path(fwd_ckpt).exists():
                fwd_controlnet = HairControlNet(
                    model_id=model_id,
                    vae=self.vae,
                    num_layers=num_layers,
                    local_files_only=local_files_only,
                )
                ckpt = torch.load(fwd_ckpt, map_location="cpu", weights_only=True)
                fwd_controlnet.load_state_dict(ckpt["controlnet"])
                for p in fwd_controlnet.parameters():
                    p.requires_grad_(False)
                fwd_controlnet.eval()
                self.forward_controlnet = fwd_controlnet
                self.accelerator.print(f"[Cycle] Loaded frozen forward ControlNet from {fwd_ckpt}")
            else:
                self.accelerator.print("[Cycle] forward_checkpoint not found — cycle loss disabled")

        # Phase 2 weight transfer (Phase 1 → Phase 2, controlnet only)
        resume_from = cfg["training"].get("resume_from")
        if resume_from and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location="cpu", weights_only=True)
            self.controlnet.load_state_dict(ckpt["controlnet"])
            self.accelerator.print(f"Loaded Phase 1 weights from {resume_from}")
            # best.pth는 EMA 기준 perceptual val로 선정되므로, EMA shadow가 있으면 그
            # 값으로 다시 덮어써 phase2 시작 가중치를 선정 근거와 일치시킨다([0724] planning §6).
            # EMAModel.state_dict()의 shadow는 named_parameters(requires_grad=True) 부분집합이라
            # strict=False로 로드(버퍼/고정 파라미터는 그대로 둠).
            if "ema" in ckpt:
                self.controlnet.load_state_dict(ckpt["ema"]["shadow"], strict=False)
                self.accelerator.print(f"Loaded Phase 1 EMA weights from {resume_from}")

        # Full resume: load controlnet weights early (optimizer init needs correct params)
        resume = cfg["training"].get("resume")
        if resume and Path(resume).exists():
            ckpt = torch.load(resume, map_location="cpu", weights_only=True)
            self.controlnet.load_state_dict(ckpt["controlnet"])

    def _setup_data(self):
        cfg = self.cfg["training"]
        aug = build_augmentation_pipeline(self.phase)

        dataset_name = cfg["dataset"]
        replay_sampler = None
        if dataset_name == "both":
            from torch.utils.data import ConcatDataset
            train_ds = ConcatDataset([
                HairRegionDataset(split="unbraid_train", augmentation=aug),
                HairRegionDataset(split="braid_train",   augmentation=aug),
            ])
            val_ds = ConcatDataset([
                HairRegionDataset(split="unbraid_test"),
                HairRegionDataset(split="braid_test"),
            ])
        elif dataset_name == "replay":
            # 서브샘플 없이 base 풀 전부 사용(unbraid 3,000 + braid 1,000). 8:8 stratified
            # sampler가 매 epoch unbraid 1,000(무작위 재표집) + braid 1,000(전부)을 뽑아
            # 배치 단위 1:1을 유지한다 ([0724] planning §5-1).
            from torch.utils.data import ConcatDataset
            ub = HairRegionDataset(split="unbraid_train", augmentation=aug)
            br = HairRegionDataset(split="braid_train",   augmentation=aug)
            train_ds = ConcatDataset([ub, br])
            self._domain_split = len(ub)       # 0..len(ub)-1 = unbraid, 이후 = braid
            val_ds = ConcatDataset([
                HairRegionDataset(split="unbraid_test"),
                HairRegionDataset(split="braid_test"),
            ])
            replay_sampler = StratifiedBatchSampler(
                n_unbraid=len(ub), n_braid=len(br), split_idx=self._domain_split,
                na=8, nb=8, seed=cfg.get("sample_seed", 0),
            )
        else:
            train_ds = HairRegionDataset(split=f"{dataset_name}_train", augmentation=aug)
            val_ds   = HairRegionDataset(split=f"{dataset_name}_test")

        bs = cfg.get("batch_size", 4)
        if replay_sampler is not None:
            self.train_loader = DataLoader(
                train_ds,
                batch_sampler=replay_sampler,
                num_workers=4,
                pin_memory=True,
            )
        else:
            self.train_loader = DataLoader(
                train_ds,
                batch_size=bs,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=2,
        )

    def _setup_perceptual_eval(self):
        """held-out 16(unbraid_test)+16(braid_test) 고정 평가셋 준비 ([0724] planning §4-1).

        train split에서 뽑지 않음 — 과적합·forgetting 진단이 목적이므로 held-out 필수.
        서버 데이터의 실제 파일명을 미리 알 수 없으므로, 정렬된 stem 목록을 고정 seed로
        셔플해 결정론적으로 선택한다(재실행해도 항상 동일한 16장이 나옴).
        """
        n = PERCEPTUAL_VAL_N
        ub_ds = HairRegionDataset(split="unbraid_test")
        br_ds = HairRegionDataset(split="braid_test")
        ub_idx = select_fixed_indices(len(ub_ds), n, seed=PERCEPTUAL_VAL_SEED)
        br_idx = select_fixed_indices(len(br_ds), n, seed=PERCEPTUAL_VAL_SEED)

        self._pval_unbraid = self._collate_fixed(ub_ds, ub_idx)
        self._pval_braid   = self._collate_fixed(br_ds, br_idx)
        self.accelerator.print(
            f"[PerceptualVal] held-out eval set: {n} unbraid_test + {n} braid_test "
            f"(seed={PERCEPTUAL_VAL_SEED})"
        )

    @staticmethod
    def _collate_fixed(ds: HairRegionDataset, indices: list[int]) -> dict:
        items = [ds[i] for i in indices]
        return {k: torch.stack([it[k] for it in items]) for k in ("sketch", "matte", "target")}

    def _setup_optimizer(self):
        cfg = self.cfg["training"]
        lr = cfg.get("learning_rate", 1e-4)

        self.optimizer = AdamW(
            self.controlnet.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )

        epochs = cfg.get("epochs", 200)
        steps_per_epoch = len(self.train_loader)
        total_steps = epochs * steps_per_epoch
        warmup = cfg.get("warmup_steps", 500)

        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup, 1),
            eta_min=1e-6,
        )
        self.warmup_steps = warmup
        self.total_steps = total_steps

    def _prepare_accelerator(self):
        (
            self.controlnet,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.controlnet,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
        )
        # VAE and transformer stay on device but are not managed by Accelerator
        device = self.accelerator.device
        self.vae = self.vae.to(device)
        self.transformer = self.transformer.to(device)

    # ------------------------------------------------------------------
    # Timestep / sigma sampling (logit-normal for flow matching)
    # ------------------------------------------------------------------

    def _sample_sigmas(self, bsz: int, device: torch.device) -> torch.Tensor:
        """
        Sample sigma values using logit-normal distribution.

        Returns:
            sigmas: (B, 1, 1, 1) suitable for broadcasting against (B, 16, 64, 64)
        """
        n_train = self.scheduler.config.num_train_timesteps
        u = torch.normal(
            mean=self.logit_mean,
            std=self.logit_std,
            size=(bsz,),
            device=device,
        )
        u = torch.sigmoid(u)  # [0, 1]
        # Map to scheduler sigma values
        indices = (u * n_train).long().clamp(0, n_train - 1)
        sigmas = self.scheduler.sigmas[indices.cpu()].to(device=device)  # (B,)
        return sigmas.view(bsz, 1, 1, 1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg["training"]
        epochs = cfg.get("epochs", 200)
        eval_every = self.cfg["checkpointing"].get("eval_every", 10)
        perceptual_every = self.cfg["checkpointing"].get("perceptual_every", 1)
        save_every = self.cfg["checkpointing"].get("save_every", 20)
        grad_clip = cfg.get("gradient_clip", 1.0)

        self.accelerator.print(
            f"Starting {self.phase} training for {epochs} epochs"
            + (f" (resuming from epoch {self.start_epoch})" if self.start_epoch else "")
        )

        # epoch-0 baseline — 학습 시작 전 상태로 perceptual val 1회 실행해 _best_unbraid의
        # 앵커로 사용. phase2 이관 직후 상태를 baseline으로 잡아야 게이트가 '이미 1 epoch
        # 열화된 값'에 앵커링되지 않는다. phase1은 미학습 모델이라 baseline이 커서 무해
        # ([0724] planning §4-3).
        pval0 = self._perceptual_validate()
        self._best_unbraid = min(self._best_unbraid, pval0["dE_unbraid"])
        self.accelerator.log({f"{k}/ep0": v for k, v in pval0.items()}, step=self.global_step)

        for epoch in range(self.start_epoch, epochs):
            self._current_epoch = epoch + 1
            self.controlnet.train()

            epoch_losses = []
            progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch in progress:
                loss, log_dict = self._train_step(batch, grad_clip=grad_clip)
                epoch_losses.append(log_dict["loss_total"])

                # Linear warmup then cosine decay
                if self.global_step < self.warmup_steps:
                    lr_scale = min(1.0, (self.global_step + 1) / max(self.warmup_steps, 1))
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.cfg["training"]["learning_rate"] * lr_scale
                else:
                    self.lr_scheduler.step()

                # EMA update on unwrapped controlnet
                self.ema.update(self.accelerator.unwrap_model(self.controlnet))
                self.global_step += 1

                progress.set_postfix({k: f"{v:.4f}" for k, v in log_dict.items()})
                self.accelerator.log(log_dict, step=self.global_step)

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.accelerator.print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

            # flow val — 참고용으로만 드물게 로깅(품질 지표가 아니므로 best.pth 저장은
            # 여기서 하지 않는다; 정점 선정은 아래 perceptual val 기준으로만 함, [0724] planning §4-3).
            if (epoch + 1) % eval_every == 0:
                val_loss = self._validate()
                self.accelerator.print(f"Val loss: {val_loss:.4f}")
                self.accelerator.log(
                    {"val_loss": val_loss, "epoch": epoch + 1},
                    step=self.global_step,
                )
                self.best_val_loss = min(self.best_val_loss, val_loss)

            stop = False
            if (epoch + 1) % perceptual_every == 0:
                pval = self._perceptual_validate()
                self.accelerator.log(pval, step=self.global_step)

                u = pval["dE_unbraid"]                                  # ↓ 좋음
                self._best_unbraid = min(self._best_unbraid, u)

                # (1) 조기종료 트리거 — unbraid '악화'(best 대비 5% 초과)가 3 epoch 연속이면 stop.
                # '미개선'이 아니라 '악화'로 판정하는 이유: phase2의 unbraid 목표는 '유지'이지
                # '개선'이 아니라서, 미개선 기준이면 거의 즉시 발동해 braid를 배우기 전에 끝난다.
                if u > self._best_unbraid * 1.05:
                    self._patience += 1
                else:
                    self._patience = 0

                # (2) 정점 ckpt 저장 — phase별 '채택 기준'으로 best.pth 직접 저장.
                # save_every 격자와 무관하게 정점을 항상 보존한다.
                if self.phase == "pretrain":                            # phase1: unbraid 품질이 목표
                    if u < self._best_select - 1e-4:
                        self._best_select = u
                        self._save_checkpoint("best.pth")
                else:                                                   # phase2: braid가 목표, unbraid 비악화 게이트
                    b = pval["lpips_braid"]                             # ↓ 좋음 (edge_iou는 보조 확인)
                    unbraid_ok = (u <= self._best_unbraid * 1.05)       # epoch-0 baseline 포함 최적 대비 5% 이내만 후보
                    if unbraid_ok and b < self._best_select - 1e-4:
                        self._best_select = b
                        self._save_checkpoint("best.pth")
                    # 게이트와 무관한 braid 최적도 참고용으로 별도 보존 (fallback 재료)
                    if b < getattr(self, "_best_braid_any", float("inf")) - 1e-4:
                        self._best_braid_any = b
                        self._save_checkpoint("best_ungated.pth")

                if self._patience >= 3:
                    self.accelerator.print(f"Early stop at epoch {epoch+1}")
                    stop = True

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pth")

            if stop:
                break

        if self.phase == "finetune" and not (self.output_dir / "best.pth").exists():
            self.accelerator.print(
                "[WARNING] best.pth was never saved — unbraid non-degradation gate never "
                "passed (forgetting too large). Check best_ungated.pth and epoch_*.pth for "
                "manual selection (no automatic fallback)."
            )

        self._save_checkpoint("final.pth")
        self.accelerator.end_training()

    def _train_step(self, batch: dict, grad_clip: float = 1.0) -> tuple[torch.Tensor, dict]:
        matte = batch["matte"]   # (B, 1, 512, 512) [0,1]

        # Forward mode: sketch → hair  |  Inverse mode: hair → sketch
        if self.inverse_mode:
            cond_image  = batch["img"]      # full hair photo as conditioning signal
            target      = batch["sketch"]   # generate sketch
            # Weight flow loss on stroke regions (non-black sketch pixels)
            stroke_mask = (target.mean(dim=1, keepdim=True) > 0.05).float()
            loss_mask   = stroke_mask
        else:
            cond_image  = batch["sketch"]   # sketch as conditioning signal
            target      = batch["target"]   # generate hair region (img × matte)
            loss_mask   = matte

        with self.accelerator.accumulate(self.controlnet):
            device = self.accelerator.device
            B = target.shape[0]

            # 1. Encode target image → latents
            with torch.no_grad():
                latents = self.vae.encode(target)  # (B, 16, 64, 64)

            # 2. Sample sigma (flow matching)
            sigmas = self._sample_sigmas(B, device)  # (B, 1, 1, 1)

            # 3. Sample noise and form noisy latent
            noise = torch.randn_like(latents)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            # 4. Spatial loss weight at latent resolution
            matte_latent = resize_matte_to_latent(loss_mask)  # (B, 1, 64, 64)

            # Cast inputs to bfloat16 for model forward passes
            noisy_latents = noisy_latents.to(dtype=torch.bfloat16)
            # SD3.5 사전학습 규약: 모델에 주는 timestep = sigma * num_train_timesteps (0~1000).
            # raw sigma(0~1)를 넘기면 frozen DiT가 모든 노이즈 레벨을 t≈0으로 인식한다.
            # sigma 원값은 노이징(위)과 x0 복원(loss_fn)에만 사용.
            timesteps_1d = sigmas.view(B).float() * self.scheduler.config.num_train_timesteps

            # 5. ControlNet forward → block_samples + null conditioning
            #    HairControlNet VAE-encodes cond_image as its conditioning signal.
            #    In inverse mode cond_image = hair photo; in forward mode = sketch.
            block_samples, null_enc_hs, null_pooled = self.controlnet(
                noisy_latent=noisy_latents,
                sketch=cond_image,
                matte=matte,
                timesteps=timesteps_1d,
            )
            block_samples = [s.to(dtype=torch.bfloat16) for s in block_samples]
            if self.schedule != "none":
                block_samples = gate_block_samples(block_samples, matte, self.schedule, gate_alpha=self.gate_alpha)
            null_enc_hs   = null_enc_hs.to(dtype=torch.bfloat16)
            null_pooled   = null_pooled.to(dtype=torch.bfloat16)

            # 6. Frozen transformer forward with ControlNet residuals
            # NOTE: do NOT use torch.no_grad() here.
            # Transformer parameters are frozen via requires_grad_(False),
            # but the computation graph must be maintained so that gradients
            # flow back through block_controlnet_hidden_states → HairControlNet.
            v_pred = transformer_forward_full_residual(
                self.transformer,
                block_samples,
                hidden_states=noisy_latents,
                encoder_hidden_states=null_enc_hs,
                pooled_projections=null_pooled,
                timestep=timesteps_1d,
                return_dict=False,
            )[0]   # (B, 16, 64, 64)

            # 7. Flow matching velocity target: v = noise - latents
            v_target = (noise - latents).to(dtype=torch.bfloat16)

            # 8. Direct supervision loss
            compute_R = (self.global_step % 100 == 0)
            total_loss, log_dict = self.loss_fn(
                v_pred=v_pred,
                v_target=v_target,
                matte_latent=matte_latent.to(dtype=torch.bfloat16),
                x_t=noisy_latents,
                sigmas=sigmas.to(dtype=torch.bfloat16),
                vae=self.vae,
                target_rgb=target,
                sketch=cond_image,   # edge loss uses this; disabled in inverse via w_edge=0.0
                matte=loss_mask,
                current_step=self.global_step,
                total_steps=self.total_steps,
                compute_R=compute_R,
            )

            controlnet_unwrapped = self.accelerator.unwrap_model(self.controlnet)

            # 9. Cycle loss: decoder head (O(1)) preferred over full-diffusion fallback
            if self._current_epoch >= self.cycle_start and self.w_cycle > 0.0:
                if (self.inverse_mode
                        and controlnet_unwrapped.sketch_decoder is not None):
                    # Decoder head cycle: block_samples → sketch_pred (no diffusion)
                    sketch_pred_cycle = controlnet_unwrapped.sketch_decoder(block_samples)
                    sketch_gt_cycle   = batch["sketch"].to(device=device, dtype=torch.float32)
                    stroke_cycle = (sketch_gt_cycle.mean(dim=1, keepdim=True) > 0.05).float()
                    cycle_loss = (
                        F.l1_loss(sketch_pred_cycle.float(), sketch_gt_cycle, reduction="none")
                        * stroke_cycle
                    ).sum() / stroke_cycle.sum().clamp(min=1)
                    total_loss = total_loss + self.w_cycle * cycle_loss
                    log_dict["loss_cycle"] = cycle_loss.item()

                elif self.forward_controlnet is not None:
                    # Full-diffusion cycle: x_0 pred → VAE decode → forward ControlNet features
                    x0_pred = noisy_latents.float() - sigmas.float() * v_pred.float()
                    sketch_pred_rgb = self.vae.decode(x0_pred.to(dtype=torch.bfloat16)).clamp(0, 1)
                    block_pred = self.forward_controlnet._get_features_impl(
                        sketch_pred_rgb, matte.to(device=device, dtype=torch.bfloat16)
                    )
                    with torch.no_grad():
                        block_gt = self.forward_controlnet._get_features_impl(
                            batch["sketch"].to(device=device, dtype=torch.bfloat16),
                            matte.to(device=device, dtype=torch.bfloat16),
                        )
                    cycle_loss = sum(
                        F.mse_loss(bp, bg.detach())
                        for bp, bg in zip(block_pred, block_gt)
                    ) / len(block_pred)
                    total_loss = total_loss + self.w_cycle * cycle_loss
                    log_dict["loss_cycle"] = cycle_loss.item()

            # 10. Sketch decoder reconstruction loss (forward mode only)
            #     Forces ControlNet block features to encode sketch structure.
            if (not self.inverse_mode
                    and self.w_sketch_dec > 0.0
                    and controlnet_unwrapped.sketch_decoder is not None):
                sketch_pred_dec = controlnet_unwrapped.sketch_decoder(block_samples)
                sketch_gt_dec   = cond_image.to(dtype=torch.float32)
                stroke_dec = (sketch_gt_dec.mean(dim=1, keepdim=True) > 0.05).float()
                l_sketch_recon = (
                    F.l1_loss(sketch_pred_dec.float(), sketch_gt_dec, reduction="none")
                    * stroke_dec
                ).sum() / stroke_dec.sum().clamp(min=1)
                total_loss = total_loss + self.w_sketch_dec * l_sketch_recon
                log_dict["loss_sketch_recon"] = l_sketch_recon.item()

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.controlnet.parameters(),
                    max_norm=grad_clip,
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss, log_dict

    @torch.no_grad()
    def _validate(self) -> float:
        self.controlnet.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            sketch = batch["sketch"]
            matte  = batch["matte"]
            target = batch["target"]

            device = self.accelerator.device
            B = target.shape[0]

            latents = self.vae.encode(target)
            sigmas = self._sample_sigmas(B, device)
            noise = torch.randn_like(latents)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            matte_latent = resize_matte_to_latent(matte)

            noisy_latents = noisy_latents.to(dtype=torch.bfloat16)
            # SD3.5 규약: timestep = sigma * num_train_timesteps (train step과 동일)
            timesteps_1d = sigmas.view(B).float() * self.scheduler.config.num_train_timesteps

            block_samples, null_enc_hs, null_pooled = self.controlnet(
                noisy_latent=noisy_latents,
                sketch=sketch,
                matte=matte,
                timesteps=timesteps_1d,
            )
            block_samples = [s.to(dtype=torch.bfloat16) for s in block_samples]
            if self.schedule != "none":
                block_samples = gate_block_samples(block_samples, matte, self.schedule, gate_alpha=self.gate_alpha)
            null_enc_hs   = null_enc_hs.to(dtype=torch.bfloat16)
            null_pooled   = null_pooled.to(dtype=torch.bfloat16)

            v_pred = transformer_forward_full_residual(
                self.transformer,
                block_samples,
                hidden_states=noisy_latents,
                encoder_hidden_states=null_enc_hs,
                pooled_projections=null_pooled,
                timestep=timesteps_1d,
                return_dict=False,
            )[0]

            v_target = (noise - latents).to(dtype=torch.bfloat16)

            _, log_dict = self.loss_fn(
                v_pred=v_pred,
                v_target=v_target,
                matte_latent=matte_latent.to(dtype=torch.bfloat16),
            )
            total_loss += log_dict["loss_total"]
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _perceptual_validate(self) -> dict:
        """held-out 16(unbraid)+16(braid) 고정셋에 EMA 가중치로 생성 후 품질 지표 측정
        ([0724] planning §4-2). 32장을 한 번에 배치 샘플링(순차 대비 대폭 단축)한다.

        scripts/eval_metrics.py의 공식(CIEDE2000 shade-정규화 평균색 ΔE, Canny edge 기반
        IoU, alex-net LPIPS)을 그대로 재사용해 최종 리포트 수치와 일치시킨다.

        Returns:
            {"dE_unbraid":.., "lpips_unbraid":.., "edge_iou_braid":.., "lpips_braid":..}
        """
        from scripts.eval_metrics import (
            _safe_mean,
            canny_edges,
            compute_delta_e_hue,
            edge_iou,
            hair_masked_lpips,
        )
        from scripts.infer_custom import decode_hair, run_sampling_batched

        cn = self.accelerator.unwrap_model(self.controlnet)
        # EMA API는 apply_to(model)/restore_to(model, original) — copy_to()는 없음(ema.py:52-62).
        # 원본 스냅샷은 호출자가 직접 떠야 한다.
        orig = {k: v.detach().clone() for k, v in cn.named_parameters() if v.requires_grad}
        was_training = cn.training
        self.ema.apply_to(cn)
        cn.eval()

        try:
            device = self.accelerator.device
            n_ub = self._pval_unbraid["sketch"].shape[0]

            sketch = torch.cat([self._pval_unbraid["sketch"], self._pval_braid["sketch"]], dim=0).to(device)
            matte  = torch.cat([self._pval_unbraid["matte"],  self._pval_braid["matte"]],  dim=0).to(device)
            target = torch.cat([self._pval_unbraid["target"], self._pval_braid["target"]], dim=0).to(device)
            B = sketch.shape[0]

            # 생성 noise seed 고정 → epoch 간 paired 비교(분산↓).
            gen = torch.Generator().manual_seed(PERCEPTUAL_VAL_SEED)
            hair_latent, _ = run_sampling_batched(
                cn, self.transformer, self.sample_scheduler, self.schedule,
                sketch, matte, PERCEPTUAL_VAL_STEPS, device, B,
                generator=gen, gate_alpha=self.gate_alpha,
            )
            pred = decode_hair(self.vae, hair_latent)   # (B,3,512,512) [0,1]

            pred_np   = _to_u8_hwc(pred)
            gt_np     = _to_u8_hwc(target)
            matte_np  = _to_u8_hw(matte)
            sketch_np = _to_u8_hwc(sketch)

            lpips_ub, de_ub = [], []
            lpips_br, iou_br = [], []
            for i in range(B):
                m = matte_np[i]
                lp = hair_masked_lpips(pred_np[i], gt_np[i], m)
                if i < n_ub:
                    lpips_ub.append(lp)
                    de_ub.append(compute_delta_e_hue(pred_np[i], gt_np[i], m, m.astype(np.float64) / 255.0))
                else:
                    lpips_br.append(lp)
                    iou_br.append(edge_iou(canny_edges(pred_np[i]), canny_edges(sketch_np[i]), m))

            return {
                "dE_unbraid":     _safe_mean(de_ub),
                "lpips_unbraid":  _safe_mean(lpips_ub),
                "lpips_braid":    _safe_mean(lpips_br),
                "edge_iou_braid": _safe_mean(iou_br),
            }
        finally:
            self.ema.restore_to(cn, orig)
            if was_training:
                cn.train()

    def _restore_training_state(self):
        """optimizer / ema / lr_scheduler / step / epoch 전체 복원 (동일 학습 재개용)."""
        resume = self.cfg["training"].get("resume")
        if not resume or not Path(resume).exists():
            return

        ckpt = torch.load(resume, map_location="cpu", weights_only=True)

        if "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
            device = self.accelerator.device
            self.ema.shadow = {k: v.to(device) for k, v in self.ema.shadow.items()}
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        self.global_step   = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        # --start_epoch으로 수동 지정 가능 (구 checkpoint에 epoch 키 없을 때)
        self.start_epoch   = (
            self.cfg["training"].get("start_epoch")
            or ckpt.get("epoch", 0)
        )

        self.accelerator.print(
            f"Resumed from {resume} "
            f"(epoch {self.start_epoch}, step {self.global_step})"
        )

    def _save_checkpoint(self, filename: str):
        if not self.accelerator.is_main_process:
            return
        controlnet_unwrapped = self.accelerator.unwrap_model(self.controlnet)
        ckpt = {
            "controlnet":   controlnet_unwrapped.state_dict(),
            "ema":          self.ema.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step":  self.global_step,
            "epoch":        self._current_epoch,
            "best_val_loss": self.best_val_loss,
            "config":       self.cfg,
        }
        save_path = self.output_dir / filename
        torch.save(ckpt, save_path)
        self.accelerator.print(f"Saved checkpoint: {save_path}")
        infer_path = self.output_dir / filename.replace(".pth", "_infer.pth")
        torch.save({"controlnet": controlnet_unwrapped.state_dict()}, infer_path)
        self.accelerator.print(f"Saved infer checkpoint: {infer_path}")
