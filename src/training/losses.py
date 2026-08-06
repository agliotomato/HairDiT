"""
Flow matching losses for SD3.5 ControlNet hair generation.

L_total = w_flow * (L_flow^eq12 / s_clamped) + w_lpips * L_lpips + w_edge * L_edge

L_flow^eq12: masked flow matching loss (paper Eq. 12, matte 가중은 선형)
  - L_flow = Σ(m̃ ⊙ (v_pred - v_target)²) / (||m̃||₁ + ε), v_target = noise - latents
  - 분모가 헤어 면적(matte L1)이라 헤어 영역 크기와 무관하게 샘플 기여가 균등
  - matte 가중 지수는 LPIPS 마스킹(선형)과 일치시킴 — soft 경계에서만 어긋나던 감독 비율 교정
    (reports/[0729]retrain_plan_v2.md §2-4)
  - matte 바깥 weight=0.0 (BLD: loss=0 outside matte)
  - scale-sync: s_raw = numel(v_pred) / (Σ matte_latent + eps), s = clamp(s_raw, s_min, s_max).
    matte에서만 계산되는 상수 배율(grad 없음) — flow 항에 1/s를 곱해 gradient 스케일을 맞춘다.
L_lpips: perceptual loss on decoded x0_pred in matte region
  - x0_pred = x_t - sigma * v_pred  (flow matching x0 recovery)
  - decoded through SD3.5 VAE
L_edge: sketch edge alignment (braid Phase 2 only)
"""

from __future__ import annotations

from typing import Optional

import kornia.filters as KF
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.utils import resize_matte_to_latent
from src.models.vae_wrapper import VAEWrapper


class FlowMatchingLoss(nn.Module):
    """
    Masked flow matching loss — paper Eq. 12, matte 가중은 선형(m̃):

        L_flow = Σ(m̃ ⊙ (v_pred - v_target)²) / (||m̃||₁ + ε)

    - 분자: 차이를 먼저 제곱한 뒤 matte로 선형 가중한다. matte를 곱한 뒤 제곱하면
      soft 경계가 m̃² 가중이 되는데, LPIPS 마스킹은 선형(pred·matte)이라 그 영역에서만
      두 항의 감독 비율이 어긋난다 → 지수를 LPIPS와 일치시킨 것
      (reports/[0729]retrain_plan_v2.md §2-4, [0727] §4-3).
      m̃=1인 헤어 내부는 m̃²=m̃이라 이 선택의 영향을 받지 않는다.
    - 분모: matte L1 norm(= 헤어 면적) — 전체 텐서 크기(B·C·H·W)가 아니라
      헤어 영역 크기로 정규화하므로 작은 헤어 샘플도 기여가 깎이지 않는다.
      HairLoss의 scale-sync(÷s)와 함께 쓰면 Σ(m̃·d²)/numel로 정리돼 mcs2의 구 정규화와
      대수적으로 동일해진다(clamp 미발동 시).
    - outside_weight > 0 이면 m̃ 대신 [m̃ + w·(1-m̃)] 을 마스크로 사용 (기본 0.0 = 논문).

    v_target = noise - latents  (flow matching velocity)
    """

    def __init__(self, outside_weight: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.outside_weight = outside_weight
        self.eps = eps

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        matte_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            v_pred:       (B, 16, 64, 64) predicted velocity
            v_target:     (B, 16, 64, 64) target velocity (noise - latents)
            matte_latent: (B, 1, 64, 64) [0, 1] matte at latent resolution

        Returns:
            loss: scalar
        """
        weight = matte_latent + self.outside_weight * (1.0 - matte_latent)  # (B, 1, 64, 64)

        # bf16 대규모 합산 오차 방지를 위해 float32로 계산 (gradient는 그대로 흐름)
        diff_sq = (v_pred.float() - v_target.float()).pow(2)                # (B, 16, 64, 64)
        return (weight.float() * diff_sq).sum() / (weight.float().sum() + self.eps)


class PerceptualLoss(nn.Module):
    """
    LPIPS perceptual loss computed on the decoded hair region (within matte).
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        self.lpips_fn = lpips.LPIPS(net=net)
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        matte: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_rgb:   (B, 3, 512, 512) in [-1, 1]
            target_rgb: (B, 3, 512, 512) in [-1, 1]
            matte:      (B, 1, 512, 512) in [0, 1]

        Returns:
            loss: scalar
        """
        pred_masked   = pred_rgb   * matte
        target_masked = target_rgb * matte
        # LPIPS operates in float32
        return self.lpips_fn(
            pred_masked.float(),
            target_masked.float(),
        ).mean()


class SketchEdgeAlignmentLoss(nn.Module):
    """
    Structural fidelity loss for braid fine-tuning.

    Penalizes regions where the sketch has strokes but the generated hair
    has no corresponding edges (strand boundaries).

    stroke_threshold is deliberately tiny, not a perceptual "is this a visible
    stroke" cutoff: the recolor pipeline (StrokeColorSampler) guarantees
    background is exactly 0 and every real stroke pixel has max-channel >=
    ~10/255 (dark-stroke floor), so any value above float noise already means
    "stroke". A higher cutoff like the old 0.1 silently drops floor-rescued
    dark strokes from edge supervision.
    """

    def __init__(self, stroke_threshold: float = 1e-3):
        super().__init__()
        self.stroke_threshold = stroke_threshold

    def forward(
        self,
        pred_rgb: torch.Tensor,
        sketch: torch.Tensor,
        matte: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_rgb: (B, 3, 512, 512) in [-1, 1]
            sketch:   (B, 3, 512, 512) in [0, 1]
            matte:    (B, 1, 512, 512) in [0, 1]

        Returns:
            loss:           scalar
            stroke_density: scalar, ‖sketch_mask‖₁ / N — 모니터링용(s가 stroke 면적을
                             못 잡는 리스크 대비, [0724] planning §2-b)
        """
        # Sketch stroke mask: any channel > threshold, intersected with matte
        sketch_mask = (sketch.max(dim=1, keepdim=True).values > self.stroke_threshold).float()
        sketch_mask = sketch_mask * matte

        # Sobel edge magnitude on predicted hair (grayscale)
        pred_gray = pred_rgb.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        pred_gray = (pred_gray + 1.0) * 0.5  # [-1,1] → [0,1]

        # spatial_gradient returns (B, C, 2, H, W)
        grad = KF.spatial_gradient(pred_gray.float())   # (B, 1, 2, H, W)
        edge_mag = grad.norm(dim=2).clamp(0, 1)        # (B, 1, H, W)

        # Penalize: stroke present AND edge absent
        loss = (sketch_mask * (1.0 - edge_mag)).mean()
        return loss, sketch_mask.mean()


class HairLoss(nn.Module):
    """
    Combined flow-matching loss for SD3.5 ControlNet training.

    Args:
        phase:             "pretrain" (unbraid) or "finetune" (braid)
        w_flow:            weight for flow matching loss (default 1.0)
        w_lpips:           weight for perceptual loss (default 0.1)
        w_edge:            weight for edge alignment loss (default 0.05)
        lpips_noise_cutoff: maximum scheduler sigma for LPIPS activation.
                            sigma is the noise coefficient of
                            x_t = (1-sigma)*x + sigma*noise, so 0.7 disables
                            LPIPS over the highest-noise 30% of the timestep
                            *range* — the PixelGen noise gate g(t)=1[t>=0.3]
                            (paper Eq. 9) under their x_t = t*x + (1-t)*eps,
                            i.e. their t equals our (1 - sigma).
                            Only read when lpips_gate == "noise".
        lpips_gate:        which LPIPS activation rule to use (run5_2 ablation).
                            "noise"  = per-sample noise gate (run5, default —
                                       every existing config keeps its behaviour)
                            "warmup" = run4's batch-level step warmup; LPIPS is
                                       applied to the *whole* batch once
                                       current_step >= lpips_warmup_frac * total_steps
        lpips_warmup_frac: fraction of total steps before LPIPS activates.
                            Only read when lpips_gate == "warmup" (pretrain only).
        scale_sync:        divide the flow term by s=clamp(numel(v_pred)/‖matte_latent‖₁, s_min, s_max)
                            to sync its gradient scale with lpips/edge (default True)
        s_min, s_max:       clamp range for s (default 20.0 / 120.0)
    """

    def __init__(
        self,
        phase: str = "pretrain",
        w_flow: float = 1.0,
        w_lpips: float = 0.1,
        w_edge: float = 0.05,
        lpips_noise_cutoff: float = 0.7,
        lpips_gate: str = "noise",
        lpips_warmup_frac: float = 0.3,
        scale_sync: bool = True,
        s_min: float = 20.0,
        s_max: float = 120.0,
    ):
        super().__init__()
        assert phase in ("pretrain", "finetune"), f"Unknown phase: {phase}"
        if lpips_gate not in ("noise", "warmup"):
            raise ValueError(f"lpips_gate must be 'noise' or 'warmup', got {lpips_gate!r}")
        self.phase = phase
        self.w_flow = w_flow
        self.w_lpips = w_lpips
        self.w_edge = w_edge
        self.lpips_noise_cutoff = lpips_noise_cutoff
        self.lpips_gate = lpips_gate
        self.lpips_warmup_frac = lpips_warmup_frac
        self.scale_sync = scale_sync
        self.s_min, self.s_max = s_min, s_max

        self.flow_loss = FlowMatchingLoss(outside_weight=0.0)
        self.perc_loss = PerceptualLoss()
        self.edge_loss = SketchEdgeAlignmentLoss()

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        matte_latent: torch.Tensor,
        # Optional for LPIPS / edge loss
        x_t: Optional[torch.Tensor] = None,
        sigmas: Optional[torch.Tensor] = None,   # (B, 1, 1, 1) for x0 recovery
        vae: Optional[VAEWrapper] = None,
        target_rgb: Optional[torch.Tensor] = None,  # (B,3,H,W) in [0,1]
        sketch: Optional[torch.Tensor] = None,
        matte: Optional[torch.Tensor] = None,
        current_step: int = 0,
        total_steps: int = 1,
        compute_R: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            v_pred:        (B, 16, 64, 64) predicted velocity
            v_target:      (B, 16, 64, 64) ground-truth velocity (noise - latents)
            matte_latent:  (B, 1, 64, 64) matte at latent scale
            x_t:           (B, 16, 64, 64) noisy latent at time sigma
            sigmas:        (B, 1, 1, 1) flow matching sigma values
            vae:           VAEWrapper for decoding x0 predictions
            target_rgb:    (B, 3, H, W) ground truth in [0, 1]
            sketch:        (B, 3, H, W) colored sketch in [0, 1]
            matte:         (B, 1, H, W) full-res matte in [0, 1]
            current_step:  current training step
            total_steps:   total training steps
            compute_R:     log gradient-norm ratio R_lpips/R_edge w.r.t. v_pred (expensive; call
                            sparingly, e.g. every 100 steps)

        Returns:
            total_loss: scalar tensor
            loss_dict:  dict with individual loss values (for logging)
        """
        # Primary flow matching loss (paper Eq. 12) — always computed, logged raw
        l_flow = self.flow_loss(v_pred, v_target, matte_latent)
        loss_dict = {"loss_flow_eq12": l_flow.item()}

        if self.scale_sync:
            # s is computed from matte only → constant scalar w.r.t. the graph (no grad)
            s_raw = v_pred.numel() / (matte_latent.detach().float().sum() + 1e-6)
            s = s_raw.clamp(self.s_min, self.s_max)
            flow_term = self.w_flow * l_flow / s
            loss_dict.update({
                "s_raw":    float(s_raw),
                "s":        float(s),
                "clamp_hi": float(s_raw > self.s_max),
                "clamp_lo": float(s_raw < self.s_min),
            })
        else:
            flow_term = self.w_flow * l_flow

        total_loss = flow_term
        loss_dict["loss_flow"] = flow_term.item()   # actual flow term used for training

        # 2026-08-05: Replace epoch-progress LPIPS warmup with PixelGen's noise
        # gate (paper Eq. 9, g(t)=1[t>=0.3]). PixelGen interpolates as
        # x_t = t*x + (1-t)*eps, so their t is the *data* coefficient and their
        # gate keeps the noise coefficient (1-t) <= 0.7. Our sigma IS the noise
        # coefficient (noisy = (1-sigma)*latents + sigma*noise), so the same
        # gate is sigma <= 0.7 — no shift conversion applies: SD3.5's shift=3.0
        # is already baked into scheduler.sigmas, and PixelGen trains with
        # timeshift=1.0 (identity). Their Table 5f puts tau=0.1 (our 0.9) at
        # "limited effect" and tau=0.6 (our 0.4) at "substantially hurts".
        # The mask is per sample, not per batch, because a batch can contain
        # multiple sigma values.
        #
        # 2026-08-06 (run5_2 ablation): run5 changed the data (density mix) and
        # this gate at the same time, so its regression cannot be attributed to
        # either one. `lpips_gate` restores run4's step-warmup as a config
        # toggle so the two variables can be separated in a 2x2 design. The default
        # stays "noise", i.e. every config written up to run5 is unaffected.
        if self.lpips_gate == "warmup":
            # run4 semantics: batch-level on/off, LPIPS over the FULL batch.
            # A None mask downstream means "no subsetting" — do not build an
            # all-True mask here, that would silently change perc_loss input.
            lpips_sample_mask = None
            lpips_active = (
                self.phase == "finetune"
                or current_step >= int(self.lpips_warmup_frac * total_steps)
            )
            loss_dict["lpips_active_fraction"] = float(lpips_active)
        else:
            if not 0.0 < self.lpips_noise_cutoff <= 1.0:
                raise ValueError(
                    f"lpips_noise_cutoff must be in (0, 1], got {self.lpips_noise_cutoff}"
                )
            lpips_sample_mask = sigmas.view(-1).float() <= self.lpips_noise_cutoff if sigmas is not None else None
            lpips_active = bool(lpips_sample_mask is not None and lpips_sample_mask.any())
            loss_dict["lpips_active_fraction"] = (
                float(lpips_sample_mask.float().mean()) if lpips_sample_mask is not None else 0.0
            )

        l_lpips = None
        l_edge = None
        edge_active = self.phase == "finetune" and sketch is not None and matte is not None
        if (lpips_active or edge_active) and vae is not None and x_t is not None and sigmas is not None:
            # Flow matching x0 recovery: x0 = x_t - sigma * v_pred
            # x0_pred must stay in the computation graph so LPIPS gradient
            # flows back through v_pred → HairControlNet.
            # VAE params are frozen (requires_grad=False) so no VAE updates occur.
            x0_pred = x_t - sigmas * v_pred            # (B, 16, 64, 64)
            pred_rgb_11 = vae.decode(x0_pred)          # (B, 3, H, W) in [-1, 1]

            if target_rgb is not None and matte is not None and lpips_active:
                target_rgb_11 = VAEWrapper.normalize(target_rgb)
                if lpips_sample_mask is None:
                    # warmup gate (run4): no per-sample subsetting.
                    l_lpips = self.perc_loss(pred_rgb_11, target_rgb_11, matte)
                else:
                    active = lpips_sample_mask
                    l_lpips = self.perc_loss(
                        pred_rgb_11[active],
                        target_rgb_11[active],
                        matte[active],
                    )
                total_loss = total_loss + self.w_lpips * l_lpips
                loss_dict["loss_lpips"] = l_lpips.item()

            # Edge loss remains active for every phase-2 sample; timestep
            # gating applies to LPIPS only.
            if edge_active:
                l_edge, stroke_density = self.edge_loss(pred_rgb_11, sketch, matte)
                total_loss = total_loss + self.w_edge * l_edge
                loss_dict["loss_edge"] = l_edge.item()
                loss_dict["stroke_density"] = float(stroke_density)

        # Gradient-norm ratio R (w.r.t. v_pred) — monitoring only, 100-step cadence
        if compute_R and lpips_active:
            g_flow = torch.autograd.grad(flow_term, v_pred, retain_graph=True)[0].norm()
            if l_lpips is not None:
                g_lp = torch.autograd.grad(self.w_lpips * l_lpips, v_pred, retain_graph=True)[0].norm()
                loss_dict["R_lpips"] = float(g_lp / (g_flow + 1e-8))
            if l_edge is not None:
                g_ed = torch.autograd.grad(self.w_edge * l_edge, v_pred, retain_graph=True)[0].norm()
                loss_dict["R_edge"] = float(g_ed / (g_flow + 1e-8))

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict
