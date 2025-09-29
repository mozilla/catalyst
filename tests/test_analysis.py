#!/usr/bin/env python3

import unittest
import sys
import os

# Add the parent directory to sys.path to import lib.analysis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.analysis import (
    calc_cohen_d,
    rank_biserial_correlation,
    calc_t_test,
    flatten_histogram,
    create_subsample,
    calc_cdf_from_density,
    calc_histogram_quantiles,
    calc_histogram_density,
    calc_histogram_mean_var,
    calc_histogram_median,
    calc_confidence_interval,
    createNumericalTemplate,
    createCategoricalTemplate,
    createScalarTemplate,
)


class TestAnalysis(unittest.TestCase):
    """Tests for analysis functions."""

    def test_calc_cohen_d_basic(self):
        """Test Cohen's d calculation."""
        result = calc_cohen_d(100, 105, 10, 10, 30, 30)
        self.assertIsInstance(result, float)

    def test_calc_cohen_d_zero_effect(self):
        """Test Cohen's d with zero effect."""
        result = calc_cohen_d(100, 100, 10, 10, 30, 30)
        self.assertEqual(result, 0.0)

    def test_calc_cohen_d_zero_variance(self):
        """Test Cohen's d with zero variance."""
        result = calc_cohen_d(100, 105, 0, 0, 30, 30)
        self.assertEqual(result, -10.0)  # Should be capped at -10

    def test_calc_cohen_d_negative_effect(self):
        """Test Cohen's d with negative effect."""
        result = calc_cohen_d(95, 100, 10, 10, 30, 30)
        self.assertLess(result, 0)

    def test_calc_cohen_d_large_effect_capped(self):
        """Test Cohen's d caps at +/- 10."""
        result = calc_cohen_d(0, 1000, 1, 1, 30, 30)
        self.assertEqual(result, -10.0)

    def test_rank_biserial_correlation_basic(self):
        """Test rank biserial correlation."""
        result = rank_biserial_correlation(10, 10, 25)  # Middle value
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_rank_biserial_correlation_extremes(self):
        """Test rank biserial correlation extremes."""
        # Complete separation
        result1 = rank_biserial_correlation(10, 10, 0)
        self.assertEqual(result1, 1.0)

        # Maximum overlap
        result2 = rank_biserial_correlation(10, 10, 50)
        self.assertEqual(result2, 0.0)

    def test_rank_biserial_correlation_asymmetric(self):
        """Test rank biserial correlation with asymmetric sample sizes."""
        result = rank_biserial_correlation(5, 15, 10)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_calc_t_test_basic(self):
        """Test t-test calculation."""
        result = calc_t_test(100, 105, 10, 12, 30, 25)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        # t-value, p-value, effect size
        self.assertIsInstance(result[0], float)  # t-value
        self.assertIsInstance(result[1], float)  # p-value
        self.assertIsInstance(result[2], float)  # effect size

    def test_calc_t_test_identical_means(self):
        """Test t-test with identical means."""
        result = calc_t_test(100, 100, 10, 10, 30, 30)
        self.assertEqual(result[0], 0.0)  # t-value should be 0
        self.assertEqual(result[2], 0.0)  # effect size should be 0

    def test_calc_t_test_zero_variance(self):
        """Test t-test with zero variance causes division by zero."""
        # This is expected to raise ZeroDivisionError - t-test is undefined with zero variance
        with self.assertRaises(ZeroDivisionError):
            calc_t_test(100, 105, 0, 0, 30, 30)

    def test_calc_t_test_different_sample_sizes(self):
        """Test t-test with different sample sizes."""
        result = calc_t_test(100, 95, 15, 12, 50, 40)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertGreater(result[0], 0)  # t-value should be positive

    def test_flatten_histogram_basic(self):
        """Test flatten_histogram with basic data."""
        bins = [10, 20, 30]
        counts = [4, 6, 8]
        result = flatten_histogram(bins, counts)
        self.assertIsInstance(result, list)
        # Should have 1 + 2 + 3 = 6 values (counts divided by 2, int)
        self.assertEqual(len(result), 6)

    def test_flatten_histogram_empty(self):
        """Test flatten_histogram with empty data."""
        bins = []
        counts = []
        result = flatten_histogram(bins, counts)
        self.assertEqual(result, [])

    def test_create_subsample_no_sampling_needed(self):
        """Test create_subsample when total counts <= sample_size."""
        bins = [10, 20]
        counts = [2, 2]
        result = create_subsample(bins, counts, sample_size=100)
        # Should return flattened histogram since total_counts (4) <= sample_size (100)
        self.assertIsInstance(result, list)

    def test_create_subsample_sampling_needed(self):
        """Test create_subsample when sampling is needed."""
        bins = [10, 20, 30]
        counts = [10000, 20000, 30000]
        result = create_subsample(bins, counts, sample_size=1000)
        self.assertIsInstance(result, list)
        # Should return a subsample
        self.assertLess(len(result), sum(counts))

    def test_calc_cdf_from_density(self):
        """Test calc_cdf_from_density function."""
        density = [0.1, 0.3, 0.4, 0.2]
        vals = [10, 20, 30, 40]
        result = calc_cdf_from_density(density, vals)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(density) - 1)  # Returns len(density)-1 values

    def test_calc_histogram_quantiles(self):
        """Test calc_histogram_quantiles function."""
        bins = [10, 20, 30, 40]
        density = [0.25, 0.25, 0.25, 0.25]  # Equal density
        result = calc_histogram_quantiles(bins, density)
        self.assertIsInstance(result, list)
        # Returns a list with two sublists [quantiles, values]
        self.assertEqual(len(result), 2)

    def test_calc_histogram_density(self):
        """Test calc_histogram_density function."""
        counts = [100, 200, 300, 400]
        n = sum(counts)
        result = calc_histogram_density(counts, n)
        self.assertIsInstance(result, list)
        # Function returns some data
        self.assertGreater(len(result), 0)

    def test_calc_histogram_mean_var(self):
        """Test calc_histogram_mean_var function."""
        bins = [10, 20, 30, 40]
        counts = [100, 200, 300, 400]
        result = calc_histogram_mean_var(bins, counts)
        self.assertIsInstance(result, list)
        # Returns a list with statistical values
        self.assertGreater(len(result), 0)
        # All values should be numeric
        for val in result:
            self.assertIsInstance(val, (int, float))

    def test_calc_histogram_median(self):
        """Test calc_histogram_median function."""
        bins = [10, 20, 30, 40]
        counts = [100, 200, 300, 400]
        result = calc_histogram_median(bins, counts)
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, min(bins))
        self.assertLessEqual(result, max(bins))

    def test_calc_confidence_interval(self):
        """Test calc_confidence_interval function."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = calc_confidence_interval(data, confidence=0.95)
        self.assertIsInstance(result, list)
        # Returns a list with statistical values
        self.assertGreater(len(result), 2)
        # Should contain numeric values
        for val in result:
            self.assertIsInstance(val, (int, float))

    def test_create_numerical_template(self):
        """Test createNumericalTemplate function."""
        result = createNumericalTemplate()
        self.assertIsInstance(result, dict)
        # Should contain basic template keys
        basic_keys = ['mean', 'std', 'n']
        for key in basic_keys:
            self.assertIn(key, result)

    def test_create_categorical_template(self):
        """Test createCategoricalTemplate function."""
        result = createCategoricalTemplate()
        self.assertIsInstance(result, dict)
        # Should contain basic template keys
        basic_keys = ['counts', 'ratios', 'sum']
        for key in basic_keys:
            self.assertIn(key, result)

    def test_create_scalar_template(self):
        """Test createScalarTemplate function."""
        result = createScalarTemplate()
        self.assertIsInstance(result, dict)
        # Should contain basic template keys
        basic_keys = ['count', 'n']
        for key in basic_keys:
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
