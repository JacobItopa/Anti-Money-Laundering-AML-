"""
Unit Tests for src/models/drift_detection.py

Tests the KS-test drift detection logic using synthetic reference/current DataFrames.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.drift_detection import detect_data_drift


@pytest.fixture
def identical_parquet(tmp_path):
    """Writes two identical parquet files for drift-free comparison."""
    df = pd.DataFrame({
        'Amount Received': np.random.lognormal(3, 1, 1000),
        'Amount Paid': np.random.lognormal(3, 1, 1000),
        'Hour': np.random.randint(0, 24, 1000),
        'DayOfWeek': np.random.randint(0, 7, 1000),
    })
    ref_path = str(tmp_path / "reference.parquet")
    curr_path = str(tmp_path / "current.parquet")
    df.to_parquet(ref_path)
    df.to_parquet(curr_path)
    return ref_path, curr_path


@pytest.fixture
def drifted_parquet(tmp_path):
    """Writes reference and a severely drifted current parquet file."""
    ref = pd.DataFrame({
        'Amount Received': np.random.normal(10, 1, 1000),
        'Amount Paid': np.random.normal(10, 1, 1000),
        'Hour': np.random.randint(0, 24, 1000),
        'DayOfWeek': np.random.randint(0, 7, 1000),
    })
    curr = pd.DataFrame({
        'Amount Received': np.random.normal(500000, 1000, 1000),  # massively shifted
        'Amount Paid': np.random.normal(500000, 1000, 1000),
        'Hour': np.random.randint(0, 24, 1000),
        'DayOfWeek': np.random.randint(0, 7, 1000),
    })
    ref_path = str(tmp_path / "reference.parquet")
    curr_path = str(tmp_path / "current.parquet")
    ref.to_parquet(ref_path)
    curr.to_parquet(curr_path)
    return ref_path, curr_path


class TestDetectDataDrift:
    def test_no_drift_on_identical_data(self, identical_parquet):
        ref, curr = identical_parquet
        result = detect_data_drift(ref, curr)
        assert result is False, "Identical distributions should not trigger a drift alert"

    def test_drift_detected_on_shifted_data(self, drifted_parquet):
        ref, curr = drifted_parquet
        result = detect_data_drift(ref, curr)
        assert result is True, "Severely shifted distributions should trigger drift alert"

    def test_returns_boolean(self, identical_parquet):
        ref, curr = identical_parquet
        result = detect_data_drift(ref, curr)
        assert isinstance(result, bool)
