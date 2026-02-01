"""Benchmark configuration."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking GNN models.
    
    This class encapsulates all hyperparameters and settings for training
    and evaluating GNN models, enabling reproducible comparisons.
    """
    
    # Model architecture
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    
    # Training
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    num_epochs: int = 200
    early_stopping_patience: Optional[int] = None
    
    # Optimization
    optimizer: str = "adam"  # 'adam', 'sgd', 'adamw'
    
    # Device
    device: str = "auto"  # 'auto', 'cpu', 'cuda', 'mps'
    
    # Reproducibility
    seed: Optional[int] = 42
    
    # Logging
    verbose: bool = True
    log_interval: int = 10
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.hidden_channels > 0, "hidden_channels must be positive"
        assert self.num_layers >= 2, "num_layers must be at least 2"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.optimizer in ["adam", "sgd", "adamw"], \
            "optimizer must be 'adam', 'sgd', or 'adamw'"
        assert self.device in ["auto", "cpu", "cuda", "mps"], \
            "device must be 'auto', 'cpu', 'cuda', or 'mps'"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        return {
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'optimizer': self.optimizer,
            'device': self.device,
            'seed': self.seed,
            'verbose': self.verbose,
            'log_interval': self.log_interval,
            'model_params': self.model_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration
            
        Returns:
            BenchmarkConfig instance
        """
        return cls(**config_dict)

    def __repr__(self) -> str:
        return f"BenchmarkConfig({self.to_dict()})"
