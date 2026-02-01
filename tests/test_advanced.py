"""Advanced tests for SimpleGNN package."""

import unittest
from sample.core import GNNModel, create_model
from sample.helpers import evaluate_model


class TestAdvancedGNNFeatures(unittest.TestCase):
    """Test cases for advanced GNN features."""

    def test_model_dimensions(self):
        """Test model with various dimension configurations."""
        test_cases = [
            (5, 32, 1),
            (10, 64, 2),
            (20, 128, 5),
            (50, 256, 10),
        ]
        for input_dim, hidden_dim, output_dim in test_cases:
            with self.subTest(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim):
                model = GNNModel(input_dim, hidden_dim, output_dim)
                self.assertEqual(model.input_dim, input_dim)
                self.assertEqual(model.hidden_dim, hidden_dim)
                self.assertEqual(model.output_dim, output_dim)


class TestModelFactoryAdvanced(unittest.TestCase):
    """Advanced test cases for model factory."""

    def test_create_model_with_kwargs(self):
        """Test creating model with keyword arguments."""
        model = create_model(
            'base',
            input_dim=15,
            hidden_dim=128,
            output_dim=3
        )
        self.assertIsInstance(model, GNNModel)
        self.assertEqual(model.input_dim, 15)
        self.assertEqual(model.hidden_dim, 128)
        self.assertEqual(model.output_dim, 3)


class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""

    def test_metrics_structure(self):
        """Test that evaluation returns proper metrics structure."""
        model = create_model('base', input_dim=10, hidden_dim=64, output_dim=2)
        metrics = evaluate_model(model, None)
        
        # Check metrics structure
        self.assertIsInstance(metrics, dict)
        required_keys = ['accuracy', 'loss']
        for key in required_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""

    def test_full_workflow(self):
        """Test the complete model creation and evaluation workflow."""
        # Step 1: Create model
        model = create_model('base', input_dim=10, hidden_dim=64, output_dim=2)
        self.assertIsNotNone(model)
        
        # Step 2: Evaluate model
        metrics = evaluate_model(model, None)
        self.assertIsNotNone(metrics)
        
        # Step 3: Verify metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('loss', metrics)


if __name__ == '__main__':
    unittest.main()
