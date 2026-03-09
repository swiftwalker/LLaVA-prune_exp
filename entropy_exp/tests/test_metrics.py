import os
import sys

import numpy as np


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from metrics import topk_concentration


def test_topk_concentration_matches_sort_when_k_lt_n():
    scores = np.array([0.2, 0.1, 0.4, 0.3], dtype=np.float64)
    got = topk_concentration(scores, k=2)
    expected = (0.4 + 0.3) / scores.sum()
    assert np.isclose(got, expected)


def test_topk_concentration_k_eq_n_returns_one_for_positive_sum():
    scores = np.array([0.2, 0.1, 0.4, 0.3], dtype=np.float64)
    got = topk_concentration(scores, k=4)
    assert np.isclose(got, 1.0)


def test_topk_concentration_k_zero_returns_zero():
    scores = np.array([0.2, 0.1, 0.4, 0.3], dtype=np.float64)
    got = topk_concentration(scores, k=0)
    assert np.isclose(got, 0.0)
