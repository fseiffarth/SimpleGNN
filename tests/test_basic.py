"""Basic tests for SimpleGNN package."""

import unittest
from sample.core import GNNModel, create_model
from sample.helpers import load_dataset, preprocess_graph, evaluate_model


class TestGNNModel(unittest.TestCase):
    """Test cases for GNNModel class."""

    def test_model_initialization(self):
        """Test that GNNModel initializes correctly."""
        model = GNNModel(input_dim=10, hidden_dim=64, output_dim=2)
        self.assertEqual(model.input_dim, 10)
        self.assertEqual(model.hidden_dim, 64)
        self.assertEqual(model.output_dim, 2)

    def test_model_forward_not_implemented(self):
        """Test that forward method raises NotImplementedError."""
        model = GNNModel(input_dim=10, hidden_dim=64, output_dim=2)
        with self.assertRaises(NotImplementedError):
            model.forward(None)


class TestModelFactory(unittest.TestCase):
    """Test cases for model factory function."""

    def test_create_base_model(self):
        """Test creating base model type."""
        model = create_model('base', input_dim=10, hidden_dim=64, output_dim=2)
        self.assertIsInstance(model, GNNModel)
        self.assertEqual(model.input_dim, 10)

    def test_create_unknown_model(self):
        """Test that creating unknown model type raises ValueError."""
        with self.assertRaises(ValueError):
            create_model('unknown_type', input_dim=10, hidden_dim=64, output_dim=2)


class TestHelpers(unittest.TestCase):
    """Test cases for helper functions."""

    def test_load_dataset(self):
        """Test loading a dataset."""
        result = load_dataset('test_dataset')
        # Should return None for placeholder implementation
        self.assertIsNone(result)

    def test_preprocess_graph(self):
        """Test graph preprocessing."""
        test_graph = {'nodes': [1, 2, 3], 'edges': [(1, 2), (2, 3)]}
        result = preprocess_graph(test_graph)
        # Should return the same graph for placeholder implementation
        self.assertEqual(result, test_graph)

    def test_evaluate_model(self):
        """Test model evaluation."""
        model = create_model('base', input_dim=10, hidden_dim=64, output_dim=2)
        metrics = evaluate_model(model, None)
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('loss', metrics)


if __name__ == '__main__':
    unittest.main()
