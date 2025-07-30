#!/usr/bin/env python3

"""
Test LSTM baseline validation framework
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.lstm_baseline import LSTMBaselineValidator, create_test_data


class TestLSTMBaseline(unittest.TestCase):
    """Test LSTM baseline validation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.validator = LSTMBaselineValidator()
        self.features, self.targets, self.data = create_test_data()

    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        self.assertIsNotNone(self.validator.config)
        self.assertIsNotNone(self.validator.lstm_builder)
        self.assertIsNotNone(self.validator.logger)

    def test_test_data_creation(self):
        """Test synthetic data creation"""
        self.assertEqual(self.features.shape, (1000, 30, 12))
        self.assertEqual(len(self.targets), 1000)
        self.assertEqual(len(self.data), 1000)

    def test_fold_metrics_calculation(self):
        """Test fold metrics calculation"""
        y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        y_pred = np.array([0.015, -0.015, 0.025, -0.005, 0.018])
        
        metrics = self.validator._calculate_fold_metrics(y_true, y_pred)
        
        self.assertIn('directional_accuracy', metrics)
        self.assertIn('correlation', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)

    def test_ensemble_readiness_assessment(self):
        """Test ensemble readiness assessment"""
        mock_results = {
            'directional_accuracy_mean': 0.55,
            'sharpe_ratio_mean': 0.9,
            'max_drawdown_mean': -0.12,
            'correlation_mean': 0.3
        }
        
        readiness = self.validator._assess_ensemble_readiness(mock_results)
        
        self.assertIn('ready_for_ensemble', readiness)
        self.assertIn('recommendation', readiness)
        self.assertTrue(readiness['ready_for_ensemble'])


if __name__ == "__main__":
    unittest.main()