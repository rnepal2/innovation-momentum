import unittest

import numpy as np

from spectral import graph_fourier_basis, spearman_corr


class SpectralHelpersTest(unittest.TestCase):
    def test_spearman_corr_orders_monotonic_vectors(self):
        self.assertAlmostEqual(spearman_corr(np.array([1, 2, 3]), np.array([10, 20, 30])), 1.0)
        self.assertAlmostEqual(spearman_corr(np.array([1, 2, 3]), np.array([30, 20, 10])), -1.0)

    def test_graph_fourier_basis_returns_sorted_eigenvalues(self):
        adjacency = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        eigvals, eigvecs = graph_fourier_basis(adjacency)
        self.assertTrue(np.all(np.diff(eigvals) >= -1e-9))
        self.assertEqual(eigvecs.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
