import numpy as np
import pytest
from correlated_hops import (
    Bath,
    concurrence,
    connected,
    partial_trace,
    solve,
    spin_chain,
    spin_operators,
    stationary_noise,
)
from correlated_hops.reference import master_equation
from scipy.linalg import expm


def test_spin_algebra():
    o = spin_operators(2)
    for i in range(2):
        np.testing.assert_allclose(
            o["sx"][i] @ o["sy"][i] - o["sy"][i] @ o["sx"][i], 1j * o["sz"][i]
        )
    np.testing.assert_allclose(o["sx"][0] @ o["sy"][1], o["sy"][1] @ o["sx"][0])


def test_nonuniform_complex_bath_covariance():
    b = Bath([[0.4, 0.2 + 0.15j], [0.2 - 0.15j, 1.2]], [0.08, 0.9])
    z = stationary_noise(b, 0.4, 1, 60000, seed=1729)
    np.testing.assert_allclose(z[0] @ z[0].conj().T / 60000, np.eye(2), atol=0.015)
    np.testing.assert_allclose(z[1] @ z[0].conj().T / 60000, b.kernel(0.4), atol=0.015)
    np.testing.assert_allclose(z[1] @ z[1].conj().T / 60000, np.eye(2), atol=0.015)
    np.testing.assert_allclose(z[1] @ z[0].T / 60000, np.zeros((2, 2)), atol=0.015)


def test_undamped_noise_is_unitary():
    b = Bath([[0.5, 0.2], [0.2, 1.0]], [0, 0])
    z = stationary_noise(b, 0.1, 2, 3)
    np.testing.assert_allclose(z[1], b.kernel(0.1) @ z[0], atol=1e-7)


def test_noise_joint_covariance_includes_nonzero_origins_and_negative_lags():
    bath = Bath([[0.4, 0.2 + 0.15j], [0.2 - 0.15j, 1.2]], [0.08, 0.9])
    z = bath.sample(0.2, 5, 30000, seed=1729)
    selected = [0, 2, 5]
    joint = z[selected].reshape(6, 30000)
    times = np.array(selected) * 0.2
    target = np.block(
        [
            [
                bath.kernel(t - s) if t >= s else bath.kernel(s - t).conj().T
                for s in times
            ]
            for t in times
        ]
    )
    np.testing.assert_allclose(joint @ joint.conj().T / 30000, target, atol=0.035)
    np.testing.assert_allclose(joint @ joint.T / 30000, np.zeros((6, 6)), atol=0.035)
    np.testing.assert_allclose(joint.mean(axis=1), np.zeros(6), atol=0.025)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_zero_coupling_recovers_schrodinger(nonlinear):
    model = spin_chain(2, g=0, J=0.4, delta=1.3)
    initial = np.array([0, 1, 0, 0], complex)
    t = np.linspace(0, 1, 6)
    r = solve(model, initial, t, dt=0.02, kmax=1, samples=2, nonlinear=nonlinear)
    for ti, rho in zip(t, r.density):
        v = expm(-1j * model.hamiltonian * ti) @ initial
        np.testing.assert_allclose(rho, np.outer(v, v.conj()), atol=1e-9)


def test_excitation_transport_matches_master_equation():
    model = spin_chain(2, g=0.5, counter_rotating=0, Jb=0.35, kappa=[0.1, 0.6])
    initial = np.array([0, 0, 1, 0], complex)
    t = np.linspace(0, 2, 11)
    r = solve(model, initial, t, dt=0.01, kmax=[1, 2], samples=3, nonlinear=False)
    exact = master_equation(model, initial, t, photon_cutoff=1)
    # In the one-excitation JC sector these excited-state entries are noise
    # independent. This tests bath hopping, complex drift and linear estimator.
    np.testing.assert_allclose(r.density[:, 1:, 1:], exact[:, 1:, 1:], atol=2e-8)
    assert exact[-1, 1, 1].real > 0.002


def test_nonlinear_rabi_agrees_with_master_equation():
    model = spin_chain(2, g=0.4, counter_rotating=0.5, Jb=0.3, kappa=[0.1, 0.6])
    initial = np.array([1, 0, 0, 0], complex)
    t = np.linspace(0, 0.8, 5)
    r = solve(model, initial, t, dt=0.02, kmax=3, samples=1024, seed=104)
    exact = master_equation(model, initial, t, photon_cutoff=3)
    np.testing.assert_allclose(np.trace(r.density, axis1=1, axis2=2), 1, atol=1e-12)
    # Fixed, predeclared stochastic tolerance, with componentwise SEM.
    projectors = np.einsum("si,sj->sij", r.roots[-1], r.roots[-1].conj())
    sem = np.std(projectors, axis=0, ddof=1) / np.sqrt(1024)
    assert np.all(abs(r.density[-1] - exact[-1]) < 5 * sem + 3e-4)
    assert np.linalg.norm(r.density[-1] - exact[-1]) < 0.015


def test_nested_time_and_hierarchy_refinement():
    model = spin_chain(2, g=0.35, Jb=0.25, kappa=[0.1, 0.4])
    t = np.linspace(0, 0.4, 3)
    z = stationary_noise(model.bath, 0.0025, 160, 2, 23)
    initial = np.array([1, 0, 0, 0], complex)
    fine = solve(model, initial, t, dt=0.005, kmax=4, samples=2, noise=z)
    coarse = solve(model, initial, t, dt=0.01, kmax=4, samples=2, noise=z[::2])
    shallower = solve(model, initial, t, dt=0.005, kmax=3, samples=2, noise=z)
    assert np.max(abs(fine.density - coarse.density)) < 3e-4
    assert np.max(abs(fine.density - shallower.density)) < 1e-6


def test_ensemble_connected_and_entanglement():
    a = np.array([1, 0, 0, 0])
    b = np.array([0, 0, 0, 1])
    rho = (np.outer(a, a) + np.outer(b, b)) / 2
    o = spin_operators(2)
    assert connected(rho, o["sz"][0], o["sz"][1]) == pytest.approx(0.25)
    bell = (a + b) / np.sqrt(2)
    assert concurrence(np.outer(bell, bell)) == pytest.approx(1, abs=1e-7)
    assert concurrence(rho) == pytest.approx(0)
    np.testing.assert_allclose(partial_trace(rho, [0], [2, 2]), np.eye(2) / 2)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Bath([[1, 0.2], [0, 1]], [0.1, 0.1])
    with pytest.raises(ValueError):
        Bath([[1]], [-0.1])
    with pytest.raises(ValueError):
        solve(spin_chain(1), [1, 0], [0, 0.13], dt=0.1)
