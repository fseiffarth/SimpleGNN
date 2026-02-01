"""Benchmark class for training and evaluating GNN models."""

import time
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from simplegnn.models.base import BaseGNN
from simplegnn.datasets.base import BaseDataset
from simplegnn.benchmark.config import BenchmarkConfig
from simplegnn.metrics import Metrics


class Benchmark:
    """Benchmark for training and evaluating GNN models.
    
    This class provides a unified interface for training GNN models on datasets
    and comparing their performance with standardized metrics.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize the benchmark.
        
        Args:
            config: Benchmark configuration. If None, uses default config.
        """
        self.config = config or BenchmarkConfig()
        self.device = self._setup_device()
        self.results: List[Dict[str, Any]] = []
        
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.seed)

    def _setup_device(self) -> torch.device:
        """Setup computation device.
        
        Returns:
            PyTorch device
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def _get_optimizer(self, model: BaseGNN) -> torch.optim.Optimizer:
        """Get optimizer for the model.
        
        Args:
            model: GNN model
            
        Returns:
            PyTorch optimizer
        """
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def train_epoch(self, model: BaseGNN, data: torch.Tensor,
                    optimizer: torch.optim.Optimizer) -> float:
        """Train model for one epoch.
        
        Args:
            model: GNN model
            data: Graph data
            optimizer: Optimizer
            
        Returns:
            Training loss
        """
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def evaluate(self, model: BaseGNN, data: torch.Tensor,
                 mask: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on a data split.
        
        Args:
            model: GNN model
            data: Graph data
            mask: Boolean mask for the split
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        out = model(data.x, data.edge_index)
        
        y_true = data.y[mask]
        y_pred = out[mask]
        
        return Metrics.compute_all(y_true, y_pred, loss=True)

    def train(self, model: BaseGNN, dataset: BaseDataset) -> Dict[str, Any]:
        """Train and evaluate a model on a dataset.
        
        Args:
            model: GNN model to train
            dataset: Dataset to train on
            
        Returns:
            Dictionary containing training results and final metrics
        """
        # Move model and data to device
        model = model.to(self.device)
        data = dataset.data.to(self.device)
        
        # Get optimizer
        optimizer = self._get_optimizer(model)
        
        # Training history
        history = {
            'train_loss': [],
            'val_metrics': [],
            'epoch_times': []
        }
        
        best_val_metric = 0
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        iterator = range(self.config.num_epochs)
        if self.config.verbose:
            iterator = tqdm(iterator, desc="Training")
        
        for epoch in iterator:
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(model, data, optimizer)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(model, data, data.val_mask)
            history['val_metrics'].append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            history['epoch_times'].append(epoch_time)
            
            # Logging
            if self.config.verbose and (epoch + 1) % self.config.log_interval == 0:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val F1: {val_metrics['f1_macro']:.4f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if self.config.early_stopping_patience is not None:
                val_metric = val_metrics['accuracy']
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        if self.config.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
        
        # Load best model if early stopping was used
        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
        
        # Final evaluation
        train_metrics = self.evaluate(model, data, data.train_mask)
        val_metrics = self.evaluate(model, data, data.val_mask)
        test_metrics = self.evaluate(model, data, data.test_mask)
        
        # Compile results
        result = {
            'model_name': model.__class__.__name__,
            'dataset_name': dataset.name,
            'config': self.config.to_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'history': history,
            'total_time': sum(history['epoch_times']),
        }
        
        self.results.append(result)
        
        if self.config.verbose:
            print("\n" + "="*60)
            print(f"Final Results for {model.__class__.__name__} on {dataset.name}")
            print("="*60)
            print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
            print(f"Test F1 (Micro): {test_metrics['f1_micro']:.4f}")
            print(f"Total Time:     {result['total_time']:.2f}s")
            print("="*60 + "\n")
        
        return result

    def compare_models(self, models: List[BaseGNN], dataset: BaseDataset) -> pd.DataFrame:
        """Compare multiple models on a dataset.
        
        Args:
            models: List of GNN models to compare
            dataset: Dataset to evaluate on
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model in models:
            if self.config.verbose:
                print(f"\nTraining {model.__class__.__name__}...")
            
            result = self.train(model, dataset)
            
            comparison_results.append({
                'Model': model.__class__.__name__,
                'Test Accuracy': result['test_metrics']['accuracy'],
                'Test F1 (Macro)': result['test_metrics']['f1_macro'],
                'Test F1 (Micro)': result['test_metrics']['f1_micro'],
                'Test Precision': result['test_metrics']['precision'],
                'Test Recall': result['test_metrics']['recall'],
                'Train Time (s)': result['total_time'],
            })
        
        df = pd.DataFrame(comparison_results)
        
        if self.config.verbose:
            print("\n" + "="*80)
            print("MODEL COMPARISON RESULTS")
            print("="*80)
            print(df.to_string(index=False))
            print("="*80 + "\n")
        
        return df

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all benchmark results.
        
        Returns:
            List of result dictionaries
        """
        return self.results

    def save_results(self, filepath: str):
        """Save benchmark results to a file.
        
        Args:
            filepath: Path to save results (supports .csv, .json)
        """
        if not self.results:
            print("No results to save.")
            return
        
        # Extract comparison data
        comparison_data = []
        for result in self.results:
            comparison_data.append({
                'Model': result['model_name'],
                'Dataset': result['dataset_name'],
                'Test Accuracy': result['test_metrics']['accuracy'],
                'Test F1 (Macro)': result['test_metrics']['f1_macro'],
                'Test F1 (Micro)': result['test_metrics']['f1_micro'],
                'Test Precision': result['test_metrics']['precision'],
                'Test Recall': result['test_metrics']['recall'],
                'Train Time (s)': result['total_time'],
            })
        
        df = pd.DataFrame(comparison_data)
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError("Filepath must end with .csv or .json")
        
        if self.config.verbose:
            print(f"Results saved to {filepath}")
