"""
가설 검증: flow loss 정규화 변경(.mean() → sum/‖matte‖₁)이 flow의 gradient를
~50배 키워서, 가중치 그대로인 LPIPS(0.1)/edge(0.05)가 사실상 무력화됐는가?

각 loss 항을 따로 backward 해서 controlnet(학습 파라미터)에 대한
'가중치 곱한 gradient L2 norm'을 측정한다. 옵티마이저가 실제로 보는 건
loss 값이 아니라 이 gradient 크기다.

- NEW flow(현재 코드: sum/matte)와 OLD flow(.mean())를 같은 forward에서 둘 다 측정 →
  모델 상태와 무관하게 renorm 배율을 직접 보여줌.
- LPIPS / edge 도 실제 파이프라인대로 측정 (phase=finetune 이라 둘 다 활성).

실행:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/debug/gradnorm_probe.py
"""
from __future__ import annotations
import os, sys, copy
os.environ.setdefault("WANDB_MODE", "disabled")
import torch

sys.path.insert(0, os.getcwd())
from scripts.train import load_config           # noqa: E402
from src.training.trainer import Trainer         # noqa: E402
from src.models.controlnet_sd35 import gate_block_samples, transformer_forward_full_residual  # noqa: E402
from src.models.vae_wrapper import VAEWrapper    # noqa: E402
from src.data.utils import resize_matte_to_latent  # noqa: E402

N_BATCHES = 4     # 평균낼 배치 수
torch.manual_seed(0)


def grad_norm(model) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            tot += p.grad.detach().float().pow(2).sum().item()
    return tot ** 0.5


def forward_vpred(tr, batch):
    """_train_step forward를 복제하되 transformer는 no_grad로 돌려 v_pred(detached) 반환.
    이후 v_pred를 leaf로 삼아 각 loss의 ‖dL/dv_pred‖를 잰다 (transformer backward 불필요)."""
    device = tr.accelerator.device
    cond_image = batch["sketch"]
    target     = batch["target"]
    matte      = batch["matte"]
    B = target.shape[0]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latents = tr.vae.encode(target)
        sigmas = tr._sample_sigmas(B, device)
        noise = torch.randn_like(latents)
        noisy = ((1.0 - sigmas) * latents + sigmas * noise).to(dtype=torch.bfloat16)
        matte_latent = resize_matte_to_latent(matte).to(dtype=torch.bfloat16)
        timesteps = sigmas.view(B).float() * tr.scheduler.config.num_train_timesteps

        block_samples, null_hs, null_pooled = tr.controlnet(
            noisy_latent=noisy, sketch=cond_image, matte=matte, timesteps=timesteps)
        block_samples = [s.to(dtype=torch.bfloat16) for s in block_samples]
        if tr.schedule != "none":
            block_samples = gate_block_samples(block_samples, matte, tr.schedule, gate_alpha=tr.gate_alpha)
        v_pred = transformer_forward_full_residual(
            tr.transformer, block_samples,
            hidden_states=noisy, encoder_hidden_states=null_hs.to(torch.bfloat16),
            pooled_projections=null_pooled.to(torch.bfloat16),
            timestep=timesteps, return_dict=False)[0]
    v_target = (noise - latents).to(dtype=torch.bfloat16)
    return dict(v_pred=v_pred.detach(), v_target=v_target, noisy=noisy, sigmas=sigmas,
                matte_latent=matte_latent, target=target, cond_image=cond_image, matte=matte)


def gnorm_leaf(v):
    return v.grad.detach().float().norm().item()


def main():
    cfg = load_config("configs/joint_phase2.yaml")
    # 실제 학습 파이프라인과 동일 조건, 단 이 머신에서 돌게 최소 조정
    cfg["training"]["phase"] = "finetune"      # LPIPS+edge 둘 다 활성 (실제 phase2 로그와 동일)
    cfg["training"]["dataset"] = "both"
    cfg["training"]["batch_size"] = 1
    cfg["training"]["resume"] = None
    cfg["training"]["resume_from"] = None      # joint ckpt 없음 → SD3.5 초기화 상태로 측정
    cfg["training"]["schedule"] = cfg["training"].get("schedule", "all")
    cfg["training"]["loss_weights"]["edge"] = 0.05   # 실제 phase2 edge on

    print(">>> Trainer 구성 중 (SD3.5 로드)...")
    tr = Trainer(cfg)
    lf = tr.loss_fn
    w_flow, w_lpips, w_edge = lf.w_flow, lf.w_lpips, lf.w_edge
    print(f">>> weights: w_flow={w_flow}, w_lpips={w_lpips}, w_edge={w_edge}")
    print(f">>> phase={lf.phase}  (finetune=LPIPS+edge active)\n")

    it = iter(tr.train_loader)
    agg = {k: [] for k in ["flow_new", "flow_old", "lpips", "edge",
                            "loss_flow_new", "loss_flow_old", "loss_lpips", "loss_edge",
                            "matte_frac"]}
    eps = 1e-6

    for bi in range(N_BATCHES):
        batch = next(it)
        f = forward_vpred(tr, batch)
        ml = f["matte_latent"].float()
        vt = f["v_target"].float()
        noisy, sig, matte = f["noisy"], f["sigmas"].to(torch.bfloat16), f["matte"]
        agg["matte_frac"].append(ml.mean().item())

        def measure(build_loss):
            """v_pred leaf에 대한 ‖dL/dv_pred‖ 와 loss 값 반환."""
            v = f["v_pred"].detach().float().requires_grad_(True)
            loss = build_loss(v)
            v.grad = None
            loss.backward()
            torch.cuda.empty_cache()
            return gnorm_leaf(v), loss.item()

        # FLOW new (현재 코드: sum/‖matte‖₁)
        g, l = measure(lambda v: w_flow * ((ml * (v - vt)).pow(2).sum() / (ml.sum() + eps)))
        agg["flow_new"].append(g); agg["loss_flow_new"].append(l / w_flow)
        # FLOW old (.mean() over full tensor)
        g, l = measure(lambda v: w_flow * (ml.expand_as(v) * (v - vt).pow(2)).mean())
        agg["flow_old"].append(g); agg["loss_flow_old"].append(l / w_flow)

        # LPIPS: x0 = noisy - sigma*v → decode → perc_loss
        def lpips_loss(v):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x0 = noisy - sig * v.to(torch.bfloat16)
                rgb = tr.vae.decode(x0)
                return w_lpips * lf.perc_loss(rgb, VAEWrapper.normalize(f["target"]), matte)
        g, l = measure(lpips_loss)
        agg["lpips"].append(g); agg["loss_lpips"].append(l / w_lpips)

        # EDGE: decode → edge_loss(pred_rgb, sketch, matte)
        def edge_loss_fn(v):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x0 = noisy - sig * v.to(torch.bfloat16)
                rgb = tr.vae.decode(x0)
                loss, _ = lf.edge_loss(rgb, f["cond_image"], matte)
                return w_edge * loss
        g, l = measure(edge_loss_fn)
        agg["edge"].append(g); agg["loss_edge"].append(l / w_edge)

        del f
        torch.cuda.empty_cache()
        print(f"  batch {bi+1}/{N_BATCHES} done")

    def mean(k): return sum(agg[k]) / len(agg[k])

    gf_new, gf_old, gl, ge = mean("flow_new"), mean("flow_old"), mean("lpips"), mean("edge")
    print("\n================ 결과 (평균, {} batches) ================".format(N_BATCHES))
    print(f"matte가 latent에서 차지하는 평균 면적 비율: {mean('matte_frac')*100:.1f}%")
    print("\n[loss 값]")
    print(f"  flow(NEW sum/matte) = {mean('loss_flow_new'):.4f}")
    print(f"  flow(OLD .mean())   = {mean('loss_flow_old'):.6f}   -> NEW/OLD 배율 = {mean('loss_flow_new')/max(mean('loss_flow_old'),1e-12):.1f}x")
    print(f"  lpips               = {mean('loss_lpips'):.4f}")
    print(f"  edge                = {mean('loss_edge'):.4f}")
    print("\n[가중치 곱한 gradient 크기  ‖d(w·L)/dv_pred‖  →  각 loss가 모델 출력을 미는 힘]")
    print(f"  w_flow ·∇flow (NEW) = {gf_new:.4e}")
    print(f"  w_flow ·∇flow (OLD) = {gf_old:.4e}")
    print(f"  w_lpips·∇lpips      = {gl:.4e}")
    print(f"  w_edge ·∇edge       = {ge:.4e}")
    print("\n[비율 = flow가 다른 항을 압도하는 배수]")
    print(f"  NEW norm:  flow/lpips = {gf_new/max(gl,1e-20):8.1f}x   flow/edge = {gf_new/max(ge,1e-20):8.1f}x")
    print(f"  OLD norm:  flow/lpips = {gf_old/max(gl,1e-20):8.1f}x   flow/edge = {gf_old/max(ge,1e-20):8.1f}x")
    print("\n해석: NEW에서 flow/lpips 비가 OLD보다 크게 뛰면, renorm이 LPIPS/edge를")
    print("      상대적으로 눌렀다는 뜻(가설 지지). 두 norm에서 비슷하면 가설 기각.")


if __name__ == "__main__":
    main()
