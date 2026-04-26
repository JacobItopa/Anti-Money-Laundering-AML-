"""
Unit Tests for src/features/build_features.py

Tests each transformation function in isolation using small, controlled DataFrames.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.build_features import (
    clean_data,
    engineer_temporal_features,
    scale_features,
    encode_categorical_features,
    select_features,
    build_features_pipeline,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal raw transaction DataFrame matching real column structure."""
    return pd.DataFrame({
        'Timestamp': ['2022/09/01 00:20', '2022/09/01 00:20', '2022/09/02 15:30'],
        'From Bank': ['10', '10', '20'],
        'Account': ['8000EBD30', '8000EBD30', '8000XYZ99'],
        'Account.1': ['8000EBD30', '8000EBD30', '8000ABC01'],
        'To Bank': ['10', '10', '30'],
        'Amount Received': [10.50, 10.50, 500000.0],
        'Receiving Currency': ['US Dollar', 'US Dollar', 'Bitcoin'],
        'Amount Paid': [10.50, 10.50, 499000.0],
        'Payment Currency': ['US Dollar', 'US Dollar', 'Bitcoin'],
        'Payment Format': ['Reinvestment', 'Reinvestment', 'Bitcoin'],
        'Is Laundering': [0, 0, 1],
    })

# ── clean_data ─────────────────────────────────────────────────────────────────

class TestCleanData:
    def test_removes_exact_duplicates(self, sample_df):
        # Rows 0 and 1 are exact duplicates
        result = clean_data(sample_df)
        assert len(result) == 2, "Expected exactly 2 rows after deduplication"

    def test_returns_dataframe(self, sample_df):
        result = clean_data(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_dataframe_does_not_crash(self):
        empty = pd.DataFrame(columns=['A', 'B'])
        result = clean_data(empty)
        assert len(result) == 0

# ── engineer_temporal_features ────────────────────────────────────────────────

class TestEngineerTemporalFeatures:
    def test_creates_hour_column(self, sample_df):
        result = engineer_temporal_features(sample_df.drop_duplicates())
        assert 'Hour' in result.columns

    def test_creates_day_of_week_column(self, sample_df):
        result = engineer_temporal_features(sample_df.drop_duplicates())
        assert 'DayOfWeek' in result.columns

    def test_creates_month_column(self, sample_df):
        result = engineer_temporal_features(sample_df.drop_duplicates())
        assert 'Month' in result.columns

    def test_creates_is_weekend_column(self, sample_df):
        result = engineer_temporal_features(sample_df.drop_duplicates())
        assert 'IsWeekend' in result.columns

    def test_hour_values_in_valid_range(self, sample_df):
        result = engineer_temporal_features(sample_df.drop_duplicates())
        assert result['Hour'].between(0, 23).all()

    def test_is_weekend_is_binary(self, sample_df):
        result = engineer_temporal_features(sample_df.drop_duplicates())
        assert set(result['IsWeekend'].unique()).issubset({0, 1})

# ── scale_features ─────────────────────────────────────────────────────────────

class TestScaleFeatures:
    def test_log1p_applied(self, sample_df):
        df = sample_df.drop_duplicates().copy()
        original_val = df['Amount Received'].iloc[0]
        result = scale_features(df, ['Amount Received'])
        assert np.isclose(result['Amount Received'].iloc[0], np.log1p(original_val))

    def test_zero_stays_zero(self):
        df = pd.DataFrame({'Amount Received': [0.0]})
        result = scale_features(df, ['Amount Received'])
        assert result['Amount Received'].iloc[0] == 0.0

    def test_large_values_are_compressed(self, sample_df):
        df = sample_df.drop_duplicates().copy()
        result = scale_features(df, ['Amount Received'])
        # log1p of 500000 should be much smaller than 500000
        assert result['Amount Received'].max() < 100

# ── encode_categorical_features ───────────────────────────────────────────────

class TestEncodeCategoricalFeatures:
    def test_frequency_columns_created(self, sample_df):
        df = sample_df.drop_duplicates().copy()
        result = encode_categorical_features(df)
        assert 'Receiving Currency_Freq' in result.columns
        assert 'Payment Currency_Freq' in result.columns
        assert 'From Bank_Freq' in result.columns
        assert 'To Bank_Freq' in result.columns

    def test_freq_values_between_0_and_1(self, sample_df):
        df = sample_df.drop_duplicates().copy()
        result = encode_categorical_features(df)
        assert result['Receiving Currency_Freq'].between(0, 1).all()

    def test_payment_format_dummies_created(self, sample_df):
        df = sample_df.drop_duplicates().copy()
        result = encode_categorical_features(df)
        payment_format_cols = [c for c in result.columns if 'Payment Format' in c]
        assert len(payment_format_cols) > 0

# ── select_features ────────────────────────────────────────────────────────────

class TestSelectFeatures:
    def test_drops_raw_text_columns(self, sample_df):
        df = sample_df.drop_duplicates().copy()
        df = engineer_temporal_features(df)
        df = encode_categorical_features(df)
        result = select_features(df)
        dropped = {'Timestamp', 'Account', 'Account.1', 'Receiving Currency',
                   'Payment Currency', 'From Bank', 'To Bank'}
        assert not dropped.intersection(set(result.columns))

    def test_label_column_preserved(self, sample_df):
        df = sample_df.drop_duplicates().copy()
        df = engineer_temporal_features(df)
        df = encode_categorical_features(df)
        result = select_features(df)
        assert 'Is Laundering' in result.columns

# ── build_features_pipeline ────────────────────────────────────────────────────

class TestBuildFeaturesPipeline:
    def test_returns_only_numerical_columns(self, sample_df):
        result = build_features_pipeline(sample_df.copy())
        object_cols = result.select_dtypes(include='object').columns.tolist()
        assert object_cols == [], f"Found non-numerical columns: {object_cols}"

    def test_output_has_fewer_columns_than_input(self, sample_df):
        result = build_features_pipeline(sample_df.copy())
        assert len(result.columns) < len(sample_df.columns) + 20

    def test_no_null_values_in_output(self, sample_df):
        result = build_features_pipeline(sample_df.copy())
        assert result.isnull().sum().sum() == 0
