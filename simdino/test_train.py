"""Numerical checks on CPUs, including two-device placement and blog collectives."""
import pickle
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange
from PIL import Image
from jax.sharding import Mesh

import train
from checkpoint import load, save
from config import Config
from data import load_views
from model import normalize
from train import Trainer, coding_rate, cross_view_alignment, ema, feature_spread


@pytest.fixture(scope="module")
def example(
):
    config = Config(size=16, patch=8, width=12, depth=1, heads=3,
                    dim=8, batch=4, steps=4, warmup=0)
    trainer = Trainer(config)
    state = trainer.init(jax.random.key(0))
    views = jax.random.normal(jax.random.key(1), (2, 4, 16, 16, 3))
    return trainer, state, views


def close_tree(
    actual,
    expected,
    atol=2e-5,
):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(x, y, atol=atol, rtol=1e-4)


def test_loss_geometry_and_teacher_boundary(
    example,
):
    trainer, state, views = example
    config = trainer.config
    z = trainer.student.apply(state.student_weights, views[0])
    np.testing.assert_allclose(np.linalg.norm(z, axis=-1), 1, atol=1e-6)
    spread = jnp.eye(8)[None]
    collapsed = jnp.tile(jnp.eye(8)[0], (1, 8, 1))
    assert coding_rate(spread, config.eps) > coding_rate(collapsed, config.eps)
    for z in (spread, collapsed):
        moment = np.asarray(z[0]).T @ np.asarray(z[0]) / 8
        matrix = np.eye(8) + 8 / config.eps**2 * moment
        expected = np.linalg.slogdet(matrix)[1] / 2
        np.testing.assert_allclose(coding_rate(z, config.eps), expected, rtol=1e-6)
    loss_fn = lambda student, teacher: trainer.loss(student, teacher, views)[0]
    gs, gt = jax.grad(loss_fn, argnums=(0, 1))(state.student_weights, state.teacher_weights)
    assert float(sum(jnp.sum(g*g) for g in jax.tree.leaves(gs))) > 0
    assert all(np.all(g == 0) for g in jax.tree.leaves(gt))
    # A global logdet is not the average of per-device logdets.
    z = normalize(jax.random.normal(jax.random.key(2), (2, 4, 8)))
    local = (coding_rate(z[:, :2], config.eps) + coding_rate(z[:, 2:], config.eps)) / 2
    assert not np.isclose(coding_rate(z, config.eps), local)


def test_alignment_matches_opposite_crops_of_the_same_image(
):
    student = jnp.eye(4).reshape(2, 2, 4)
    teacher = student[::-1]
    np.testing.assert_allclose(cross_view_alignment(student, teacher), 0)
    np.testing.assert_allclose(cross_view_alignment(student, student), 1)
    np.testing.assert_allclose(cross_view_alignment(student, teacher[:, ::-1]), 1)
    np.testing.assert_allclose(cross_view_alignment(student, -teacher), 2)
    collapsed = jnp.ones((2, 3, 4)) / 2
    np.testing.assert_allclose(feature_spread(collapsed), 0)


@pytest.mark.parametrize("mode", ["dp", "fsdp"])
def test_sharded_loss_gradients_and_updates_match_single_device(
    example,
    mode,
):
    if jax.device_count() < 2:
        pytest.skip("Two CPU devices required")
    trainer, state, views = example
    reference = jax.jit(trainer.step)
    sharded, step, placement = trainer.compile(state, mode)
    for _ in range(2):
        state, expected = reference(state, views)
        sharded, actual = step(sharded, jax.device_put(views, placement))
        close_tree(expected, actual)
        close_tree(state, sharded)
    if mode == "fsdp":
        assert not sharded.student_weights["embed"].is_fully_replicated


def test_ema_checkpoint_and_deterministic_views(
    example,
    tmp_path,
):
    trainer, state, views = example
    updated, _ = trainer.step(state, views)
    expected_teacher = ema(state.teacher_weights, updated.student_weights, trainer.config.momentum)
    close_tree(updated.teacher_weights, expected_teacher)
    save(tmp_path / "last.pkl", updated, trainer.config)
    restored_config, restored = load(tmp_path / "last.pkl")
    assert restored_config == trainer.config
    close_tree(trainer.step(restored, views), trainer.step(updated, views))
    image = tmp_path / "sample.jpg"
    pixels = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image)
    a, b = load_views((image, 42), trainer.config.size), load_views((image, 42), trainer.config.size)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2, 16, 16, 3) and a.dtype == np.uint8
    assert not np.array_equal(a[0], a[1])


def test_blog_explicit_collectives_match_global_loss_and_gradients(
    example,
):
    if jax.device_count() < 2:
        pytest.skip("Two CPU devices required")
    trainer, state, views = example
    post = Path(__file__).resolve().parents[1] / '_drafts/scaling-4-sim-dino.md'
    blocks = re.findall(r"```python\n(.*?)```", post.read_text(), re.S)
    mesh = Mesh(np.array(jax.devices()), ('data',))
    # The committed draft still uses einops for its explicit-collective example.
    namespace = dict(vars(train), trainer=trainer, mesh=mesh, rearrange=rearrange)
    for block in blocks:
        if block.startswith('def global_rate') or 'def explicit_loss(' in block:
            exec(block, namespace)
    actual = namespace['explicit_loss_and_grads'](state.student_weights, state.teacher_weights, views)
    loss_fn = lambda student: trainer.loss(student, state.teacher_weights, views)[0]
    expected = jax.jit(jax.value_and_grad(loss_fn))(state.student_weights)
    close_tree(actual, expected)


def test_bf16_forward_keeps_loss_statistics_in_fp32(
    example,
):
    from dataclasses import replace
    trainer, _, views = example
    mixed = Trainer(replace(trainer.config, bf16=True))
    state = mixed.init(jax.random.key(0))
    new_state, metrics = jax.jit(mixed.step)(state, views)
    assert metrics['loss'].dtype == jnp.float32
    assert all(np.isfinite(value).all() for value in jax.tree.leaves((new_state, metrics)))
