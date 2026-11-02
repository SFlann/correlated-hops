"""Object workflow against independent dynamics and known ensemble states."""

import numpy as np
import pytest
from correlated_hops import Result, solve, spin_chain, spin_operators, stationary_noise
from correlated_hops.reference import master_equation


def test_model_workflow_and_prescribed_noise_compatibility():
    model = spin_chain(2, g=0.4, counter_rotating=0, Jb=0.3, kappa=[0.1, 0.4])
    times = np.linspace(0, 1, 6)
    initial = [0, 0, 1, 0]
    noise = model.bath.sample(0.005, 200, 3, seed=17)
    np.testing.assert_array_equal(
        noise, stationary_noise(model.bath, 0.005, 200, 3, 17)
    )
    original = noise.copy()
    result = model.solve(
        initial, times, dt=0.01, kmax=2, samples=3, nonlinear=False, noise=noise
    )
    functional = solve(
        model, initial, times, dt=0.01, kmax=2, samples=3, nonlinear=False, noise=noise
    )
    np.testing.assert_array_equal(result.roots, functional.roots)
    np.testing.assert_array_equal(noise, original)
    ops = spin_operators(2)
    receiver = ops["sp"][1] @ ops["sm"][1]
    reference = master_equation(model, initial, times, photon_cutoff=1)
    # Receiver occupation is noise independent in this linear one-excitation sector.
    np.testing.assert_allclose(
        result.expectation(receiver), reference[:, 1, 1], atol=2e-8
    )
    assert result.diagnostics["noise_source"] == "prescribed"


def test_result_observables_use_the_mixed_ensemble():
    # Both individual product states have zero connected correlation;
    # the classical mixture has Gzz=1/4 and zero concurrence.
    roots = np.array([[[1, 0, 0, 0], [0, 0, 0, 1]]], dtype=complex)
    result = Result(np.array([0.0]), roots, {})
    ops = spin_operators(2)
    assert result.connected(*ops["sz"])[0] == pytest.approx(0.25)
    assert result.concurrence()[0] == pytest.approx(0.0)
    np.testing.assert_allclose(result.partial_trace([0], [2, 2])[0], np.eye(2) / 2)


def test_object_workflow_preserves_validation():
    model = spin_chain(1)
    with pytest.raises(ValueError, match="samples"):
        model.bath.sample(0.1, 2, True)
    with pytest.raises(ValueError, match="distinct"):
        model.solve([1, 0], [0, 0.01], dt=0.1)
