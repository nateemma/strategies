import numpy as np

from research.nnmt.markov_filter import forward_filter, transition_matrix


def _posteriors(n, k, seed=0):
    p = np.random.default_rng(seed).random((n, k)) + 0.05
    return p / p.sum(axis=1, keepdims=True)


def test_memoryless_chain_returns_raw_posterior():
    pi = np.array([0.2, 0.5, 0.3])
    T = np.tile(pi, (3, 1))  # no persistence: the filter has nothing to add
    p = _posteriors(50, 3)
    np.testing.assert_allclose(forward_filter(p, T, pi), p, atol=1e-9)


def test_filter_is_causal():
    pi = np.array([0.3, 0.7])
    T = np.array([[0.95, 0.05], [0.02, 0.98]])
    p = _posteriors(200, 2, seed=1)
    full = forward_filter(p, T, pi)
    np.testing.assert_allclose(forward_filter(p[:120], T, pi), full[:120])


def test_sticky_chain_damps_a_one_bar_spike():
    pi = np.array([0.5, 0.5])
    T = np.array([[0.99, 0.01], [0.01, 0.99]])
    p = np.tile([0.9, 0.1], (20, 1))
    p[10] = [0.1, 0.9]
    f = forward_filter(p, T, pi)
    assert f[10, 1] < 0.5 < p[10, 1]


def test_transition_matrix_ignores_unknown_and_pair_boundaries():
    T, pi = transition_matrix([np.array([0, 0, 1, -1, 1]), np.array([1, 1])], 2)
    # transitions: 0->0, 0->1 (pair 1; 1->-1 and -1->1 dropped), 1->1 (pair 2); never pair1 end -> pair2 start
    np.testing.assert_allclose(T, [[0.5, 0.5], [0.0, 1.0]])
    np.testing.assert_allclose(pi, [2 / 6, 4 / 6])
