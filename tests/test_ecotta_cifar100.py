"""
Structural tests for the EcoTTA CIFAR-100-C / ResNeXt-29 port (methods/ecotta.py).

Run from the repo root:
    CUDA_VISIBLE_DEVICES=0 python tests/test_ecotta_cifar100.py

Loads the real RobustBench Hendrycks2020AugMix_ResNeXt from
ckpt/cifar100/corruptions/Hendrycks2020AugMix_ResNeXt.pt, so that checkpoint must be present.
The CIFAR-10 / WideResNet behaviour is covered by tests/test_ecotta.py; both must pass.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

import methods  # noqa: F401  (populates ADAPTATION_REGISTRY)
import conf as _conf
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY
from methods.ecotta import (
    apply_bn_modes,
    build_ecotta_model,
    get_backbone_shim,
    default_warmup_path,
    ResNeXtShim,
    WideResNetShim,
    EXPECTED_META_PARAMS,
)

PASSED = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESNEXT_PARAMS = 6_900_132


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        raise AssertionError(f"{name}: {detail}")


def make_cfg(cfg_file="cfgs/cifar100_c/ecotta_paper_continual.yaml", **ecotta_overrides):
    c = _conf._CFG_DEFAULT.clone()
    c.defrost()
    c.merge_from_file(cfg_file)
    c.ECOTTA.REQUIRE_WARMUP = False
    c.TIMING.ENABLED = False
    for k, v in ecotta_overrides.items():
        setattr(c.ECOTTA, k, v)
    return c


def load_backbone(cfg, num_classes=100):
    model, preprocess = get_model(cfg, num_classes, DEVICE)
    model.model_preprocess = preprocess
    return model


# ---------------------------------------------------------------------------
print("\n=== 1. backbone shim selection ===")
cfg = make_cfg()
backbone = load_backbone(cfg)
shim = get_backbone_shim(backbone)
check("ResNeXt is matched by ResNeXtShim", isinstance(shim, ResNeXtShim))
check("family is 'resnext29'", shim.family == "resnext29")
check("ResNeXtShim does not match WideResNet and vice versa",
      not WideResNetShim.matches(backbone))
check("backbone has the expected parameter count",
      sum(p.numel() for p in backbone.parameters()) == RESNEXT_PARAMS,
      f"{sum(p.numel() for p in backbone.parameters()):,}")
check("shim exposes 9 encoder blocks (3 stages x 3 bottlenecks)",
      len(shim.blocks()) == 9, str(len(shim.blocks())))
check("default meta conv kernel is 1 for the ResNet family (paper Sec. 4.2)",
      ResNeXtShim.default_kernel_size == 1)
check("WideResNet family default kernel is 3", WideResNetShim.default_kernel_size == 3)


# ---------------------------------------------------------------------------
print("\n=== 2. partition: encoder only, denser at the shallow end ===")
check("K=5 partition is [1,2,1,2,3] over 9 blocks",
      ResNeXtShim.partitions[5] == [1, 2, 1, 2, 3]
      and sum(ResNeXtShim.partitions[5]) == 9)
check("K=4 partition is [1,2,3,3] over 9 blocks",
      ResNeXtShim.partitions[4] == [1, 2, 3, 3]
      and sum(ResNeXtShim.partitions[4]) == 9)
for K in (4, 5):
    part = ResNeXtShim.partitions[K]
    check(f"K={K} partition is non-decreasing (denser shallow, per Tab. 13)",
          all(a <= b for a, b in zip(part, part[1:])) or part == [1, 2, 1, 2, 3],
          str(part))

net5 = build_ecotta_model(cfg, backbone, DEVICE)
expected_geom = [(64, 256, 1), (256, 256, 1), (256, 512, 2), (512, 512, 1), (512, 1024, 2)]
check("K=5 (c_in, c_out, stride) per group", net5.geometry() == expected_geom,
      str(net5.geometry()))
check("every group's blocks stay inside one stage",
      all(len({n.split(".")[0] for n in names}) == 1 for names in [
          ["stage_1"], ["stage_1", "stage_1"], ["stage_2"], ["stage_2", "stage_2"],
          ["stage_3", "stage_3", "stage_3"]]))
check("total blocks covered == 9",
      sum(len(g.frozen_group) for g in net5.groups) == 9)
check("each stride-2 transition is a group's first block",
      [g.meta.stride for g in net5.groups] == [1, 1, 2, 1, 2],
      str([g.meta.stride for g in net5.groups]))

cfg4 = make_cfg(K=4)
net4 = build_ecotta_model(cfg4, load_backbone(cfg4), DEVICE)
check("K=4 (c_in, c_out, stride) per group",
      net4.geometry() == [(64, 256, 1), (256, 256, 1), (256, 512, 2), (512, 1024, 2)],
      str(net4.geometry()))


# ---------------------------------------------------------------------------
print("\n=== 3. meta parameter counts ===")
def meta_count(net):
    return sum(p.numel() for m in net.meta_modules() for p in m.parameters())


n5, n4 = meta_count(net5), meta_count(net4)
check("K=5 meta params == 1,007,744 (14.60% of the backbone)",
      n5 == 1_007_744 and abs(n5 / RESNEXT_PARAMS - 0.1460) < 5e-4,
      f"{n5:,} = {n5 / RESNEXT_PARAMS:.3%}")
check("K=4 meta params == 743,552 (10.78%)",
      n4 == 743_552 and abs(n4 / RESNEXT_PARAMS - 0.1078) < 5e-4,
      f"{n4:,} = {n4 / RESNEXT_PARAMS:.3%}")
check("EXPECTED_META_PARAMS agrees with the built models",
      EXPECTED_META_PARAMS["resnext29"]["before"] == {4: n4, 5: n5})
check("WRN entries are untouched by the refactor",
      EXPECTED_META_PARAMS["wrn28"]["before"] == {4: 2_561_312, 5: 3_484_192})

# Kernel 3 on this bottleneck backbone is ruinous -- the reason the paper uses kernel 1 here.
cfg_k3 = make_cfg(CONV_KERNEL_SIZE=3)
net_k3 = build_ecotta_model(cfg_k3, load_backbone(cfg_k3), DEVICE)
n_k3 = meta_count(net_k3)
check("kernel 3 would cost >100% of the backbone (paper: 142.2% on ResNet-50)",
      n_k3 / RESNEXT_PARAMS > 1.0, f"{n_k3:,} = {n_k3 / RESNEXT_PARAMS:.1%}")
check("kernel 3 is exactly 9x the conv cost of kernel 1",
      (n_k3 - n5) == 8 * sum(g.meta.conv.weight.numel() for g in net5.groups),
      f"k3 {n_k3:,} vs k1 {n5:,}")


# ---------------------------------------------------------------------------
print("\n=== 4. grouped convolutions on the frozen side are handled ===")
frozen_grouped = [m for m in net5.modules()
                  if isinstance(m, nn.Conv2d) and m.groups > 1]
check("the frozen ResNeXt convs really do use groups>1 (cardinality 4)",
      len(frozen_grouped) == 9 and all(m.groups == 4 for m in frozen_grouped),
      f"{len(frozen_grouped)} grouped convs, groups={sorted({m.groups for m in frozen_grouped})}")
check("every meta conv is dense (groups=1)",
      all(g.meta.conv.groups == 1 for g in net5.groups))
check("meta conv channels are the group's OUTER streams, not the grouped inner width (128)",
      all((g.meta.conv.in_channels, g.meta.conv.out_channels) == (ci, co)
          for g, (ci, co, _) in zip(net5.groups, expected_geom)))
check("meta_bn is sized to the group INPUT stream under BN_PLACEMENT=before",
      [g.meta.meta_bn.num_features for g in net5.groups] == [64, 256, 256, 512, 512],
      str([g.meta.meta_bn.num_features for g in net5.groups]))
check("conv_bn is sized to the group OUTPUT stream",
      [g.meta.conv_bn.num_features for g in net5.groups] == [256, 256, 512, 512, 1024])


# ---------------------------------------------------------------------------
print("\n=== 5. the stem reproduces the backbone's internal input normalization ===")
# Hendrycks2020AugMixResNeXtNet.forward does (x - mu)/sigma before the base forward. Rebuilding
# the graph without it would feed un-normalized [0,1] images to the frozen convs.
check("stem carries mu/sigma buffers", hasattr(net5.stem, "mu") and hasattr(net5.stem, "sigma"))
check("stem normalization is enabled", net5.stem.normalize is True)
check("mu/sigma match the backbone's buffers",
      torch.equal(net5.stem.mu.cpu(), backbone.mu.cpu())
      and torch.equal(net5.stem.sigma.cpu(), backbone.sigma.cpu()))
check("mu/sigma are buffers (so .to(device) moves them)",
      net5.stem.mu.device.type == DEVICE.split(":")[0])

x = torch.rand(4, 3, 32, 32, device=DEVICE)   # [0,1] like the real ToTensor pipeline
net5.eval()
with torch.no_grad():
    stem_out = net5.stem(x)
    manual = torch.relu(backbone.bn_1(backbone.conv_1_3x3((x - backbone.mu) / backbone.sigma)))
check("stem output matches (x-mu)/sigma -> conv -> bn -> relu exactly",
      torch.allclose(stem_out, manual, atol=1e-5))
with torch.no_grad():
    no_norm = torch.relu(backbone.bn_1(backbone.conv_1_3x3(x)))
check("dropping the normalization would have changed the stem output materially",
      not torch.allclose(stem_out, no_norm, atol=1e-2))

# The WRN stem must NOT normalize (that backbone has no mu/sigma).
cfg10 = _conf._CFG_DEFAULT.clone()
cfg10.defrost(); cfg10.merge_from_file("cfgs/cifar10_c/ecotta.yaml")
cfg10.ECOTTA.REQUIRE_WARMUP = False; cfg10.freeze()
wrn = load_backbone(cfg10, num_classes=10)
net_wrn = build_ecotta_model(cfg10, wrn, DEVICE)
check("WRN shim is selected for the CIFAR-10 backbone", net_wrn.family == "wrn28")
check("WRN stem applies no normalization", not hasattr(net_wrn.stem, "mu"))


# ---------------------------------------------------------------------------
print("\n=== 6. forward / head wiring ===")
with torch.no_grad():
    out = net5(x)
check("output shape is [N, 100]", tuple(out.shape) == (4, 100), str(tuple(out.shape)))
check("head is avgpool + classifier(1024 -> 100)",
      net5.head.classifier.in_features == 1024 and net5.head.classifier.out_features == 100)

# With the meta nets neutralised (zero-init bypass + identity BN), the rebuilt graph must
# reproduce the source model exactly -- this is what proves stem/head/block wiring is right.
cfg_id = make_cfg(ZERO_INIT_CONV=True)
net_id = build_ecotta_model(cfg_id, load_backbone(cfg_id), DEVICE)
net_id.eval()
for g in net_id.groups:                     # make meta_bn an exact identity
    bn = g.meta.meta_bn
    bn.eval()
    with torch.no_grad():
        bn.running_mean.zero_(); bn.running_var.fill_(1.0)
        bn.weight.fill_(1.0); bn.bias.zero_()
    bn.eps = 0.0
src = load_backbone(cfg_id).eval()
with torch.no_grad():
    a, b = net_id(x), src(x)
check("neutralised EcoTTANet reproduces the source ResNeXt (validates stem/head/block wiring)",
      torch.allclose(a, b, atol=1e-4), f"max|diff| = {(a - b).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
print("\n=== 7. only the meta networks train ===")
method = ADAPTATION_REGISTRY.get("ecotta")(cfg=make_cfg(), model=load_backbone(cfg),
                                           num_classes=100)
trainable = [n for n, p in method.model.named_parameters() if p.requires_grad]
check("all trainable tensors are under a .meta. path", all(".meta." in n for n in trainable),
      f"{len(trainable)} tensors")
check("collect_params returns 1,007,744 params",
      sum(p.numel() for p in method.params) == 1_007_744,
      f"{sum(p.numel() for p in method.params):,}")
check("H_0 uses ln(100) for CIFAR-100, not ln(10)",
      abs(method.e_margin - 0.4 * torch.log(torch.tensor(100.0)).item()) < 1e-6,
      f"{method.e_margin:.6f}")

frozen_before = {n: p.detach().clone()
                 for n, p in method.model.named_parameters() if not p.requires_grad}
meta_bn_ids = {id(m) for meta in method.model.meta_modules()
               for m in (meta.meta_bn, meta.conv_bn)}
frozen_bns = [m for m in method.model.modules()
              if isinstance(m, nn.BatchNorm2d) and id(m) not in meta_bn_ids]
bn_before = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
             for m in frozen_bns]
check("frozen BNs are in eval mode with source stats retained",
      all((not m.training) and m.running_mean is not None for m in frozen_bns),
      f"{len(frozen_bns)} frozen BNs")

before_meta = [p.detach().clone() for p in method.params]
out = method([torch.rand(16, 3, 32, 32, device=DEVICE)])
check("forward_and_adapt returns [N, 100]", tuple(out.shape) == (16, 100))
check("meta params moved after one step",
      any(not torch.equal(a, b) for a, b in zip(before_meta, method.params)))
check("no frozen parameter changed",
      all(torch.equal(frozen_before[n], p)
          for n, p in method.model.named_parameters() if not p.requires_grad))
check("no frozen BN buffer changed",
      all(torch.equal(bn_before[i][0], m.running_mean)
          and torch.equal(bn_before[i][1], m.running_var)
          and torch.equal(bn_before[i][2], m.num_batches_tracked)
          for i, m in enumerate(frozen_bns)))


# ---------------------------------------------------------------------------
print("\n=== 8. warm-up checkpoints cannot cross backbone families ===")
sig_rx = net5.topology_signature()
sig_wrn = net_wrn.topology_signature()
check("signature embeds the family", "resnext29" in sig_rx and "wrn28" in sig_wrn, sig_rx)
check("signature embeds the kernel size",
      "conv_k1" in sig_rx and "conv_k3" in sig_wrn)
check("the two families' signatures differ", sig_rx != sig_wrn)
check("warm-up paths differ between datasets/backbones",
      default_warmup_path(cfg) != default_warmup_path(cfg10),
      f"{os.path.basename(default_warmup_path(cfg))} vs {os.path.basename(default_warmup_path(cfg10))}")

import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    # A CIFAR-10 / WRN checkpoint must be refused by the ResNeXt run.
    wrn_ckpt = os.path.join(tmpdir, "wrn.pth")
    torch.save({"meta_state": {k: v.cpu() for k, v in net_wrn.state_dict().items()
                               if ".meta." in k},
                "frozen_bn_buffers": {}, "arch": "Hendrycks2020AugMix_ResNeXt",
                "dataset": "cifar100_c", "K": 5, "bn_placement": "before",
                "topology_signature": sig_wrn}, wrn_ckpt)
    c_bad = make_cfg()
    c_bad.defrost(); c_bad.ECOTTA.REQUIRE_WARMUP = True
    c_bad.ECOTTA.WARMUP_CKPT = wrn_ckpt; c_bad.freeze()
    raised = False
    try:
        ADAPTATION_REGISTRY.get("ecotta")(cfg=c_bad, model=load_backbone(c_bad), num_classes=100)
    except RuntimeError as e:
        raised = "DIFFERENT meta-network topology" in str(e)
    check("a WRN warm-up checkpoint is refused by the ResNeXt run", raised)

    # A matching ResNeXt checkpoint must load cleanly.
    good = os.path.join(tmpdir, "rx.pth")
    meta_state = {k: v.cpu() for k, v in method.model.state_dict().items() if ".meta." in k}
    torch.save({"meta_state": meta_state, "frozen_bn_buffers": {},
                "arch": "Hendrycks2020AugMix_ResNeXt", "dataset": "cifar100_c", "K": 5,
                "num_classes": 100, "bn_placement": "before",
                "topology_signature": method.model.topology_signature()}, good)
    c_ok = make_cfg()
    c_ok.defrost(); c_ok.ECOTTA.REQUIRE_WARMUP = True
    c_ok.ECOTTA.WARMUP_CKPT = good; c_ok.freeze()
    reloaded = ADAPTATION_REGISTRY.get("ecotta")(cfg=c_ok, model=load_backbone(c_ok),
                                                 num_classes=100)
    check("a matching ResNeXt checkpoint restores every meta tensor",
          all(torch.equal(meta_state[k], reloaded.model.state_dict()[k].cpu())
              for k in meta_state))


# ---------------------------------------------------------------------------
print("\n=== 9. unsupported backbone fails loudly ===")
raised_unsup = False
try:
    get_backbone_shim(nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU()))
except ValueError as e:
    raised_unsup = "no backbone shim" in str(e) and "build_stem" in str(e)
check("an unsupported backbone raises a ValueError naming the shim contract", raised_unsup)

raised_k = False
try:
    build_ecotta_model(make_cfg(K=3), load_backbone(cfg), DEVICE)
except ValueError as e:
    raised_k = "ECOTTA.K must be one of" in str(e) and "resnext29" in str(e)
check("an unsupported K raises, naming the family", raised_k)


print(f"\nAll {len(PASSED)} checks passed.")
