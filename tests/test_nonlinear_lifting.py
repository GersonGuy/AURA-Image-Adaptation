import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import cfg
from methods.aura import AURA, NonlinearCorrection


def test_nonlinear_correction_init_zero():
    """Verify that delta(z) = 0 and dz/dtheta = 0 at init when zero_init_output=True."""
    corr = NonlinearCorrection(latent_dim=500, param_dim=1024, h=256, activation="gelu", zero_init_output=True)
    z = torch.randn(500)
    out = corr(z)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), f"Expected zero output at init, got max {out.abs().max()}"

    # Jacobian at init should also be 0
    J = torch.func.jacrev(corr)(z)
    assert torch.allclose(J, torch.zeros_like(J), atol=1e-7), f"Expected zero Jacobian at init, got norm {J.norm()}"


def test_nonlinear_correction_activations():
    """Verify all supported activations work without error."""
    for act in ["gelu", "relu", "leaky_relu", "tanh"]:
        corr = NonlinearCorrection(latent_dim=64, param_dim=128, h=32, activation=act, zero_init_output=False)
        z = torch.randn(64)
        out = corr(z)
        assert out.shape == (128,)


def test_linear_path_unchanged():
    """Verify reconstruct_params with correction_tuple=None is identical to z @ A + phi."""
    A = torch.randn(4, 6)
    phi = torch.randn(1, 6)
    z = torch.randn(4)
    expected = z @ A + phi.squeeze(0)

    result = AURA._reconstruct_params(
        z,
        (A,),
        (phi,),
        (("w",),),
        ((torch.Size([6]),),),
        (4,),
        ((6,),),
        correction_tuple=None,
    )
    diff = (result["w"] - expected).abs().max().item()
    assert diff < 1e-6, f"Linear path changed! max diff: {diff}"


def test_nonlinear_residual_path():
    """Verify reconstruct_params with correction_tuple is phi + z @ A + delta(z)."""
    A = torch.randn(4, 6)
    phi = torch.randn(1, 6)
    z = torch.randn(4)

    corr = NonlinearCorrection(4, 6, h=8, zero_init_output=False)
    result = AURA._reconstruct_params(
        z,
        (A,),
        (phi,),
        (("w",),),
        ((torch.Size([6]),),),
        (4,),
        ((6,),),
        correction_tuple=(corr,),
    )
    expected = phi.squeeze(0) + z @ A + corr(z)
    diff = (result["w"] - expected).abs().max().item()
    assert diff < 1e-6, f"Nonlinear residual path mismatch! max diff: {diff}"


def test_config_parsing():
    """Verify YACS config parses AURA.LIFTING correctly."""
    test_cfg = cfg.clone()
    test_cfg.defrost()
    test_cfg.AURA.LIFTING.TYPE = "nonlinear"
    test_cfg.AURA.LIFTING.ACTIVATION = "relu"
    test_cfg.AURA.LIFTING.HIDDEN_DIMS = [256]
    test_cfg.freeze()

    assert test_cfg.AURA.LIFTING.TYPE == "nonlinear"
    assert test_cfg.AURA.LIFTING.ACTIVATION == "relu"
    assert test_cfg.AURA.LIFTING.HIDDEN_DIMS == [256]


if __name__ == "__main__":
    test_nonlinear_correction_init_zero()
    test_nonlinear_correction_activations()
    test_linear_path_unchanged()
    test_nonlinear_residual_path()
    test_config_parsing()
    print("All unit tests passed successfully.")

