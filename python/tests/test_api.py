"""Public API contracts and failure behaviour independent of physics tolerances."""

import numpy as np
import pytest
from correlated_hops import (
    Bath,
    Model,
    solve,
    spin_chain,
    spin_operators,
    stationary_noise,
)
from correlated_hops.reference import master_equation


@pytest.mark.parametrize(
    "times, dt", [([0, 1e-12, 0.01], 0.01), ([0, 1e-12], 0.01), ([0, 1.5e-12], 1e-12)]
)
def test_off_grid_times_are_rejected_at_any_time_scale(times, dt):
    with pytest.raises(ValueError, match="distinct internal grid steps"):
        solve(spin_chain(1), [1, 0], times, dt=dt, samples=1)


def test_sparse_output_grid_preserves_labels_and_noise_input():
    model = spin_chain(1, g=0)
    times = np.array([0, 0.1 + 0.2, 0.5])
    noise = stationary_noise(model.bath, 0.05, 10, 2)
    original = noise.copy()
    result = solve(model, [1, 0], times, dt=0.1, samples=2, kmax=1, noise=noise)
    np.testing.assert_array_equal(result.times, times)
    np.testing.assert_array_equal(noise, original)
    assert result.roots.shape == (3, 2, 2)
    assert result.density.shape == (3, 2, 2)
    assert result.diagnostics["noise_source"] == "prescribed"


@pytest.mark.parametrize(
    "value", [1.0, 1.5, True, np.bool_(True), np.nan, np.inf, 0, -1]
)
def test_integer_parameters_fail_consistently(value):
    model = spin_chain(1)
    calls = [
        lambda: spin_operators(value),
        lambda: solve(model, [1, 0], [0, 0.01], samples=value),
        lambda: solve(model, [1, 0], [0, 0.01], samples=1, kmax=value),
        lambda: stationary_noise(model.bath, 0.01, 1, value),
        lambda: master_equation(model, [1, 0], [0, 0.01], photon_cutoff=value),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="must be an integer"):
            call()


def test_numpy_integers_and_zero_noise_steps_are_supported():
    model = spin_chain(np.int64(1))
    z = stationary_noise(model.bath, 0.01, np.int64(0), np.int64(2))
    assert z.shape == (1, 1, 2)
    result = solve(model, [1, 0], [0, 0.01], samples=np.int64(1), kmax=[np.int64(1)])
    assert result.roots.shape == (2, 1, 2)


@pytest.mark.parametrize("steps", [1.0, True, np.nan, -1])
def test_noise_grid_requires_a_nonnegative_integer_step_count(steps):
    with pytest.raises(ValueError, match="steps must be an integer"):
        stationary_noise(spin_chain(1).bath, 0.01, steps, 1)


@pytest.mark.parametrize("times", [[], 0, [0, 0], [0, -0.1], [0, np.nan]])
def test_reference_rejects_invalid_time_grids(times):
    with pytest.raises(ValueError, match="equally spaced grid"):
        master_equation(spin_chain(1), [1, 0], times)


def test_model_inputs_are_defensive_copies_and_read_only():
    h = np.array([[1.0]])
    loss = np.array([0.2])
    bath = Bath(h, loss)
    H = np.diag([0.0, 1.0])
    F = np.array([[0, 1], [0, 0]], complex)
    model = Model(H, (F,), bath)
    h[:] = loss[:] = H[:] = F[:] = 7
    assert bath.h[0, 0] == 1
    assert bath.kappa[0] == 0.2
    assert model.hamiltonian[1, 1] == 1
    assert model.couplings[0][0, 1] == 1
    for a in (bath.h, bath.kappa, model.hamiltonian, *model.couplings):
        with pytest.raises(ValueError, match="read-only"):
            a.flat[0] = 0


@pytest.mark.parametrize("shape", [(3, 1, 1), (2, 1, 2)])
def test_prescribed_noise_shape_is_checked(shape):
    with pytest.raises(ValueError, match="noise must have shape"):
        solve(spin_chain(1), [1, 0], [0, 0.01], samples=2, noise=np.zeros(shape))
