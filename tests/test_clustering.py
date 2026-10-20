import unittest

import numpy as np
from scipy.sparse import csr_matrix

from analysis.clustering import ClusteringEngine


class ClusteringEngineTests(unittest.TestCase):
    def test_optimal_k_is_limited_by_document_count(self):
        matrix = csr_matrix(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        engine = ClusteringEngine()

        optimal_k = engine.find_optimal_k(matrix)

        self.assertEqual(engine.k_values, [2])
        self.assertEqual(optimal_k, 2)
        self.assertTrue(np.isfinite(engine.silhouette_scores[0]))

    def test_two_documents_use_fallback_when_silhouette_is_undefined(self):
        matrix = csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        engine = ClusteringEngine()

        optimal_k = engine.find_optimal_k(matrix)

        self.assertEqual(engine.k_values, [2])
        self.assertEqual(optimal_k, 2)
        self.assertTrue(np.isnan(engine.silhouette_scores[0]))
        self.assertEqual(len(engine.fit(matrix)), 2)

    def test_degenerate_clusters_do_not_break_optimal_k_selection(self):
        matrix = csr_matrix([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        engine = ClusteringEngine()

        optimal_k = engine.find_optimal_k(matrix)

        self.assertEqual(optimal_k, 2)
        self.assertTrue(all(np.isnan(score) for score in engine.silhouette_scores))

    def test_clustering_requires_at_least_two_documents(self):
        engine = ClusteringEngine()

        with self.assertRaisesRegex(ValueError, "At least 2 documents"):
            engine.find_optimal_k(csr_matrix([[1.0, 0.0]]))

    def test_pca_pads_results_when_only_one_feature_is_available(self):
        matrix = csr_matrix([[1.0], [0.0]])
        engine = ClusteringEngine()

        coordinates = engine.reduce_dimensions(matrix)

        self.assertEqual(coordinates.shape, (2, 2))
        np.testing.assert_array_equal(coordinates[:, 1], np.zeros(2))


if __name__ == "__main__":
    unittest.main()
