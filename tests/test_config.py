import pytest

from mse.config import BACKBONES, STRENGTH_ORDER, backbone, faithful


def test_faithful_defaults():
    c = faithful()
    assert c.fast_iterations == 5
    assert c.meta_horizon_H == 2
    assert c.k_init == 2 and c.k_max == 3
    assert c.frontier_size == 3
    assert c.frontier_weights == (1.0, 0.5, 0.25)
    assert len(c.frontier_weights) == c.frontier_size
    assert c.p_cross == 0.2 and c.l_same == 3 and c.l_cross == 2
    assert c.backbone.name == "gemma-4-31b"
    assert c.backbone.provider == "mock"  # offline by default


def test_with_backbone_is_immutable():
    c = faithful()
    c2 = c.with_backbone(provider="openai", name="gpt-5.5")
    assert c2.backbone.provider == "openai" and c2.backbone.name == "gpt-5.5"
    assert c.backbone.provider == "mock"  # original untouched (frozen copy)


def test_backbone_registry():
    assert backbone("gemma-4-e4b").provider == "mlx"
    assert backbone("gpt-5.5").provider == "openai"
    assert backbone("mock").provider == "mock"
    with pytest.raises(KeyError):
        backbone("does-not-exist")


def test_strength_order_is_registry_subset():
    assert set(STRENGTH_ORDER).issubset(set(BACKBONES))
    assert STRENGTH_ORDER[0] == "gemma-4-e2b"  # weakest first
