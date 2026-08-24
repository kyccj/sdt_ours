"""M0 smoke test for spike_reg -- no dataset, no distributed, one GPU.

Checks, in order:
  1. how many neurons get wrapped and which ones are skipped
  2. R is finite and > 0, and the loss it produces is finite under autocast
  3. backward runs and the reg term alone produces gradient on parameters
  4. the modified backward really does reach neurons that did not fire
  5. the loss-ratio controller and the brake do what they claim, on synthetic numbers
  6. time and memory cost of the regularization, measured against reg off

Run:  python smoke_spike_reg.py --model spikformer12_512 --batch 8
"""

import argparse
import time

import torch

import spike_reg
import spikformer


def make_args(**kw):
    ns = argparse.Namespace(
        reg_spike=True, reg_sc_rate="one", reg_alpha=7.0, reg_rho=5.8e-4,
        reg_start_ep=0, reg_lam_max=1e-4, reg_growth_cap=1.5,
        reg_brake_floor=0.0, reg_brake_ep=30, reg_brake_decay=0.5, reg_skip="lif",
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def build(model_name, sc_rate, device):
    model = spikformer.__dict__[model_name](kd=False).to(device)
    cfg = spike_reg.SpikeRegConfig.from_args(make_args(reg_sc_rate=sc_rate))
    names = spike_reg.convert_multispike(model, cfg, verbose=False)
    return model, cfg, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spikformer12_512")
    ap.add_argument("--batch", default=8, type=int)
    ap.add_argument("--img", default=224, type=int)
    ap.add_argument("--sc_rate", default="one", choices=["one", "wta_rev"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # ---------------------------------------------------------------- 1. wrapping
    model, cfg, names = build(args.model, args.sc_rate, device)
    n_total = sum(1 for m in model.modules()
                  if type(m).__name__ in ("Multispike", "Multispike_first"))
    print("[1] wrapped {} of {} neurons".format(len(names), n_total))
    print("    first: {}".format(names[:3]))
    print("    last : {}".format(names[-3:]))
    unwrapped = [n for n, m in model.named_modules()
                 if type(m).__name__ == "Multispike"
                 and not isinstance(getattr(model, n.split(".")[0], None),
                                    spike_reg.EIPMultispike)
                 and ".inner" not in n]
    print("    NOT wrapped: {}".format(unwrapped))

    x = torch.randn(args.batch, 3, args.img, args.img, device=device)
    target = torch.randint(0, 1000, (args.batch,), device=device)
    ce = torch.nn.CrossEntropyLoss()

    # ---------------------------------------------------------------- 2. forward
    spike_reg.REG.enabled = True
    spike_reg.REG.lam = 0.0
    model.train()
    spike_reg.REG.reset()
    with torch.cuda.amp.autocast():
        out = model(x)
        task = ce(out, target)
        R = spike_reg.REG.total()
        spikes = spike_reg.REG.spikes()
    print("[2] layers={} R={:.6g} task_loss={:.6g} spikes/img={:.6g}".format(
        spike_reg.REG.n_layers, float(R), float(task), float(spikes) / args.batch))
    assert torch.isfinite(R) and float(R) > 0, "R must be finite and positive"
    # <= not ==: in the "large" variant (choice != "base") every MS_Block owns a `lif`
    # module that its forward never calls, so those wrapped neurons contribute nothing
    assert 0 < spike_reg.REG.n_layers <= len(names)
    if spike_reg.REG.n_layers < len(names):
        print("    note: {} wrapped neurons were never called (unused MS_Block.lif "
              "in the non-base variant)".format(len(names) - spike_reg.REG.n_layers))

    # what lambda would the controller pick from these numbers?
    lam0 = cfg.rho * float(task) / float(R)
    print("    lambda for rho={:.3g} would be {:.4g}  (reg share = {:.3g})".format(
        cfg.rho, lam0, lam0 * float(R) / float(task)))

    # ---------------------------------------------------------------- 3. backward
    model.zero_grad(set_to_none=True)
    (lam0 * R).backward()
    g = [p.grad for p in model.parameters() if p.grad is not None]
    gn = sum(float(p.norm()) for p in g)
    print("[3] reg-only backward: {} tensors got gradient, total norm {:.4g}".format(
        len(g), gn))
    assert len(g) > 0 and gn > 0, "the reg term produced no gradient"

    # ---------------------------------------------------------------- 4. silent neurons
    # The point of the modified backward: a neuron whose input is far below threshold
    # still receives gradient from the reg term.  Compare against plain L2, where
    # d||x||/dx = x/||x|| is exactly zero there.
    probe = torch.zeros(1, 4, 8, 8, device=device, requires_grad=True)
    with torch.no_grad():
        probe[0, 0, 0, 0] = 4.0          # one neuron fires, the rest are silent
    probe.requires_grad_(True)
    spk = torch.floor(torch.clamp(probe, 0, 4) + 0.5) / 4.0
    one = torch.tensor(1.0, device=device)
    nrm = spike_reg._L2NormWtaRev.apply(spk, one)
    nrm.backward()
    # gradient w.r.t. the neuron OUTPUT (before the surrogate) is what we control:
    # recompute directly since floor() blocks the path
    spk2 = torch.zeros(1, 4, 8, 8, device=device, requires_grad=True)
    with torch.no_grad():
        spk2[0, 0, 0, 0] = 1.0
    spk2.requires_grad_(True)
    spike_reg._L2NormWtaRev.apply(spk2, one).backward()
    plain = torch.zeros_like(spk2, requires_grad=True)
    with torch.no_grad():
        plain[0, 0, 0, 0] = 1.0
    plain.requires_grad_(True)
    torch.linalg.vector_norm(plain).backward()
    silent_wta = float(spk2.grad[0, 1, 0, 0])
    silent_l2 = float(plain.grad[0, 1, 0, 0])
    print("[4] gradient on a silent neuron: wta_rev {:.4g} vs plain L2 {:.4g}".format(
        silent_wta, silent_l2))
    assert silent_wta != 0.0 and silent_l2 == 0.0, "the modified backward is not active"

    # ---------------------------------------------------------------- 5. controller
    cfg_b = spike_reg.SpikeRegConfig.from_args(
        make_args(reg_rho=1e-3, reg_growth_cap=1.5, reg_brake_floor=0.22, reg_brake_ep=30))
    ctl = spike_reg.LossRatioController(cfg_b)
    print("[5] controller trace (task_loss 5.0, R falling, spikes collapsing):")
    R_seq = [1000.0, 800.0, 400.0, 100.0, 20.0, 5.0]
    S_seq = [100.0, 90.0, 60.0, 25.0, 12.0, 8.0]
    for ep, (r, s) in enumerate(zip(R_seq, S_seq)):
        lam, info = ctl.step(ep, 5.0, r, s)
        print("    ep{} R={:7.1f} S/S1={:5.3f} -> lambda={:.4g} brake={:.0f}".format(
            ep, r, info["s_ratio"], lam, info["brake_on"]))
    assert info["brake_on"] == 1.0, "the floor should have fired on the collapsing run"

    # growth cap alone
    ctl2 = spike_reg.LossRatioController(
        spike_reg.SpikeRegConfig.from_args(make_args(reg_rho=1e-3, reg_growth_cap=1.5)))
    lams = []
    for ep, (r, s) in enumerate(zip(R_seq, S_seq)):
        lam, _ = ctl2.step(ep, 5.0, r, s)
        lams.append(lam)
    ratios = [b / a for a, b in zip(lams[1:-1], lams[2:]) if a > 0]
    print("    growth cap: per-epoch lambda ratios {}".format(
        ["{:.3f}".format(r) for r in ratios]))
    assert all(r <= 1.5 + 1e-6 for r in ratios), "growth cap not enforced"

    # ---------------------------------------------------------------- 6. cost
    if device == "cuda":
        def run(reg_on, iters=5):
            spike_reg.REG.enabled = reg_on
            spike_reg.REG.lam = lam0 if reg_on else 0.0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(iters):
                model.zero_grad(set_to_none=True)
                spike_reg.REG.reset()
                with torch.cuda.amp.autocast():
                    o = model(x)
                    loss = ce(o, target)
                    if reg_on:
                        loss = loss + spike_reg.REG.lam * spike_reg.REG.total()
                loss.backward()
            torch.cuda.synchronize()
            return (time.time() - t0) / iters, torch.cuda.max_memory_allocated() / 2 ** 20

        run(False, iters=2)                       # warmup
        t_off, m_off = run(False)
        t_on, m_on = run(True)
        print("[6] reg off: {:.3f}s/iter {:.0f} MiB".format(t_off, m_off))
        print("    reg on : {:.3f}s/iter {:.0f} MiB   (+{:.0f}% time, +{:.0f} MiB)".format(
            t_on, m_on, 100 * (t_on / t_off - 1), m_on - m_off))

    print("\nall checks passed")


if __name__ == "__main__":
    main()
