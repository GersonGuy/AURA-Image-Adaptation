"""
Architecture-introspection helpers shared by the TTA methods.

The methods that select *parts* of a network (AURA's stage grouping, SGD_Base's
stage unfreezing) originally hard-coded WRN / torchvision-ResNet module names.
This module centralizes the name-matching so that adding a backbone family
(ViT here) touches one file instead of every method.
"""
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

# All affine normalization layers a TTA method may adapt.
# CNNs contribute BatchNorm, ViT/Swin contribute LayerNorm, ResNet-GN GroupNorm.
NORM_LAYER_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)

# Depth ordering used by AURA / SGD_Base. "fc" is the deepest "stage".
STAGE_ORDER = ["stage_1", "stage_2", "stage_3", "stage_4", "fc"]


def is_affine_norm(module: nn.Module) -> bool:
    """A normalization layer with learnable weight/bias (nothing to adapt otherwise)."""
    return isinstance(module, NORM_LAYER_TYPES) and getattr(module, "weight", None) is not None


def unwrap_backbone(model: nn.Module):
    """
    Strip the wrappers this repo puts around a backbone and return
    ``(core, normalize, post)``:

      * ``core``      — the actual backbone module (WRN, ResNet, timm ViT, ...)
      * ``normalize`` — input-normalization module to run before ``core`` (or None)
      * ``post``      — output layer to run after ``core`` (ImageNet-X class mask, or None)

    Handles ``normalize_model`` (``nn.Sequential(normalize, model)``) and
    ``ImageNetXWrapper`` (copies the Sequential's ``__dict__`` and adds
    ``masking_layer``). Module references are returned, not names, so callers
    that need parameter names should keep using ``model.named_parameters()``.
    """
    core = model
    normalize = None
    post = getattr(model, "masking_layer", None)

    # ImageNetXWrapper / normalize_model both expose `.normalize` and `.model`
    if hasattr(core, "model") and isinstance(getattr(core, "model"), nn.Module):
        normalize = getattr(core, "normalize", None)
        core = core.model
    elif isinstance(core, nn.Sequential) and len(core) >= 2:
        normalize = core[0]
        core = core[1]

    return core, normalize, post


def is_vit(core: nn.Module) -> bool:
    """True for a timm VisionTransformer-style backbone (patch embed + block list)."""
    return hasattr(core, "blocks") and hasattr(core, "patch_embed")


def vit_stage_block_ranges(num_blocks: int, num_stages: int = 4):
    """
    Split the transformer block list into ``num_stages`` contiguous groups, so
    ViTs expose the same ``stage_1..stage_4`` interface as the CNN backbones.
    Remainder blocks go to the deepest stages (e.g. 12 blocks -> 3/3/3/3,
    10 blocks -> 2/2/3/3).
    """
    if num_blocks < num_stages:
        raise ValueError(f"Cannot split {num_blocks} transformer blocks into {num_stages} stages.")
    base, rem = divmod(num_blocks, num_stages)
    sizes = [base] * num_stages
    for i in range(rem):
        sizes[num_stages - 1 - i] += 1
    ranges, start = [], 0
    for size in sizes:
        ranges.append(range(start, start + size))
        start += size
    return ranges


def vit_stage_modules(core: nn.Module, num_stages: int = 4):
    """
    Map ``stage_1..stage_N`` / ``fc`` to lists of ViT submodules.
    Stage 1 additionally owns the patch embedding and the pre-block norm;
    the deepest stage owns the final norm(s) that feed the classifier.
    """
    ranges = vit_stage_block_ranges(len(core.blocks), num_stages)
    mapping = {}
    for i, block_range in enumerate(ranges):
        modules = [core.blocks[j] for j in block_range]
        if i == 0:
            for extra in ("patch_embed", "norm_pre"):
                if hasattr(core, extra):
                    modules.append(getattr(core, extra))
        if i == num_stages - 1:
            for extra in ("norm", "fc_norm"):
                if hasattr(core, extra):
                    modules.append(getattr(core, extra))
        mapping[f"stage_{i + 1}"] = modules

    classifier = get_classifier_module(core)
    if classifier is not None:
        mapping["fc"] = [classifier]
    return mapping


def get_classifier_name(model: nn.Module, num_classes: int) -> str:
    """
    Name (relative to ``model``) of the classifier Linear layer.

    Resolution order:
      1. the ``nn.Linear`` whose ``out_features == num_classes`` (unambiguous case)
      2. the name advertised by timm's ``pretrained_cfg["classifier"]`` — needed
         when a class mask sits behind the head (ImageNet-R/A/V2/D109, where
         ``num_classes`` != ``head.out_features``)
      3. the last ``nn.Linear`` in module order
    """
    linear_names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not linear_names:
        raise ValueError("No nn.Linear layer found — cannot locate the classifier.")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            return name

    core, _, _ = unwrap_backbone(model)
    cfg_classifier = getattr(core, "pretrained_cfg", {}).get("classifier") if hasattr(core, "pretrained_cfg") else None
    if cfg_classifier:
        prefix = ""
        for name, module in model.named_modules():
            if module is core:
                prefix = f"{name}." if name else ""
                break
        candidate = f"{prefix}{cfg_classifier}"
        if candidate in linear_names:
            logger.info(f"Classifier resolved via timm pretrained_cfg: '{candidate}' "
                        f"(out_features != num_classes={num_classes}; a class mask is likely applied)")
            return candidate

    logger.warning(f"No nn.Linear with out_features == {num_classes}; "
                   f"falling back to the last Linear layer '{linear_names[-1]}'.")
    return linear_names[-1]


def get_classifier_module(core: nn.Module):
    """The classifier submodule of a backbone (timm ``classifier`` key, else ``fc``/``head``)."""
    cfg_classifier = getattr(core, "pretrained_cfg", {}).get("classifier") if hasattr(core, "pretrained_cfg") else None
    for attr in ([cfg_classifier] if cfg_classifier else []) + ["head", "fc", "classifier"]:
        module = getattr(core, attr, None)
        if isinstance(module, nn.Linear):
            return module
    return None


def disable_fused_attention(model: nn.Module) -> int:
    """
    Switch timm attention blocks from ``F.scaled_dot_product_attention`` to the
    explicit matmul+softmax path.

    Required for AURA on transformers: the fused SDPA kernels implement neither
    forward-mode AD (``jacfwd``) nor a vmap rule, so the Jacobian of the
    observation function raises ``NotImplementedError``. Returns the number of
    attention modules switched.
    """
    count = 0
    for module in model.modules():
        if getattr(module, "fused_attn", False):
            module.fused_attn = False
            count += 1
    if count:
        logger.info(f"Disabled fused attention (F.scaled_dot_product_attention) in {count} attention "
                    f"modules — required for torch.func Jacobians through a transformer.")
    return count


def build_stage_matchers(core: nn.Module):
    """
    Build ``{stage_name: predicate(param_name) -> bool}`` used to collect the
    normalization parameters belonging to each stage.

    Predicates take a *parameter* name as produced by ``model.named_parameters()``
    (i.e. including any wrapper prefix such as ``model.``) and match on path
    segments, so wrappers do not break the mapping.
    """
    if is_vit(core):
        return _vit_stage_matchers(core)
    return _cnn_stage_matchers(core)


def _cnn_stage_matchers(core: nn.Module):
    # "stage_N" covers torchvision ResNets (layerN.), RobustBench WRN (blockN.)
    # and Hendrycks2020AugMix_ResNeXt (stage_N.).
    tokens = {
        "stage_1": ("layer1", "block1", "stage_1"),
        "stage_2": ("layer2", "block2", "stage_2"),
        "stage_3": ("layer3", "block3", "stage_3"),
        "stage_4": ("layer4", "block4"),
    }
    matchers = {}
    for stage, toks in tokens.items():
        if stage == "stage_4" and not (hasattr(core, "layer4") or hasattr(core, "block4")):
            # WRN-28-10 / ResNeXt-29 have only three stages
            continue
        matchers[stage] = (lambda toks: lambda name: any(t in name.split(".") for t in toks))(toks)
    return matchers


def _vit_stage_matchers(core: nn.Module, num_stages: int = 4):
    ranges = vit_stage_block_ranges(len(core.blocks), num_stages)
    # Norm layers that live outside the block list, assigned to the nearest stage.
    extras = {i: () for i in range(num_stages)}
    extras[0] = ("patch_embed", "norm_pre")
    extras[num_stages - 1] = ("norm", "fc_norm")

    def make_matcher(block_idxs, extra_segments):
        def match(name: str) -> bool:
            segments = name.split(".")
            if "blocks" in segments:
                pos = segments.index("blocks")
                if pos + 1 < len(segments) and segments[pos + 1].isdigit():
                    return int(segments[pos + 1]) in block_idxs
                return False
            return any(seg in extra_segments for seg in segments)
        return match

    return {
        f"stage_{i + 1}": make_matcher(set(block_range), extras[i])
        for i, block_range in enumerate(ranges)
    }
