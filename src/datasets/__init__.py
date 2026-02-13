"""Dataset module for DDE operator learning."""

from .dde_dataset import DDEDataset, create_dataloaders

__all__ = ["DDEDataset", "create_dataloaders"]
